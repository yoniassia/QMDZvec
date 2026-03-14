"""Unit tests for memclawz.enrichment module."""
import pytest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

from memclawz.enrichment import (
    enrich_memory, compute_validity_end, is_memory_expired,
    _enrich_with_gemini, _enrich_with_openai, _default_enrichment,
    _validate_type, _validate_weight, _validate_tags, _validate_validity_hours, _validate_triples,
    _generate_default_title, _parse_enrichment_response, _build_enrichment_prompt
)


@pytest.mark.unit
class TestEnrichmentValidation:
    """Test validation functions."""
    
    def test_validate_type_valid(self):
        """Test valid memory types."""
        valid_types = ["decision", "fact", "event", "insight", "preference", "relationship", "procedure"]
        for valid_type in valid_types:
            assert _validate_type(valid_type) == valid_type
    
    def test_validate_type_invalid(self):
        """Test invalid memory type defaults to 'fact'."""
        assert _validate_type("invalid") == "fact"
        assert _validate_type("") == "fact"
        assert _validate_type(None) == "fact"
    
    def test_validate_weight_valid(self):
        """Test valid weight values."""
        assert _validate_weight(0.5) == 0.5
        assert _validate_weight(0.0) == 0.0
        assert _validate_weight(1.0) == 1.0
        assert _validate_weight("0.8") == 0.8
    
    def test_validate_weight_clamping(self):
        """Test weight clamping to [0,1] range."""
        assert _validate_weight(-0.5) == 0.0
        assert _validate_weight(1.5) == 1.0
        assert _validate_weight(10) == 1.0
    
    def test_validate_weight_invalid(self):
        """Test invalid weight defaults to 0.5."""
        assert _validate_weight("invalid") == 0.5
        assert _validate_weight(None) == 0.5
        assert _validate_weight([]) == 0.5
    
    def test_validate_tags_valid(self):
        """Test valid tags processing."""
        tags = ["python", "api", "test"]
        result = _validate_tags(tags)
        assert result == tags
    
    def test_validate_tags_truncation(self):
        """Test tags truncation (max 5 tags, 20 chars each)."""
        long_tags = [f"tag{i}" for i in range(10)]  # 10 tags
        result = _validate_tags(long_tags)
        assert len(result) == 5
        
        long_tag = "a" * 30
        result = _validate_tags([long_tag])
        assert len(result[0]) == 20
    
    def test_validate_tags_invalid(self):
        """Test invalid tags input."""
        assert _validate_tags("not_a_list") == []
        assert _validate_tags(None) == []
        assert _validate_tags([None, "", "valid"]) == ["valid"]
    
    def test_validate_validity_hours_valid(self):
        """Test valid validity hours."""
        assert _validate_validity_hours(24) == 24
        assert _validate_validity_hours("48") == 48
        assert _validate_validity_hours(None) is None
    
    def test_validate_validity_hours_clamping(self):
        """Test validity hours clamping (1 hour to 1 year max)."""
        assert _validate_validity_hours(0) == 1  # minimum 1 hour
        assert _validate_validity_hours(-5) == 1
        assert _validate_validity_hours(9000) == 8760  # maximum 1 year
    
    def test_validate_validity_hours_invalid(self):
        """Test invalid validity hours."""
        assert _validate_validity_hours("invalid") is None
        assert _validate_validity_hours([]) is None
    
    def test_validate_triples_valid(self):
        """Test valid triples processing."""
        triples = [
            {"subject": "API", "predicate": "has", "object": "endpoint"},
            {"subject": "test", "predicate": "covers", "object": "functionality"}
        ]
        result = _validate_triples(triples)
        assert len(result) == 2
        assert result[0]["subject"] == "API"
    
    def test_validate_triples_truncation(self):
        """Test triples truncation (max 10, 50 chars each field)."""
        many_triples = [{"subject": f"s{i}", "predicate": f"p{i}", "object": f"o{i}"} for i in range(15)]
        result = _validate_triples(many_triples)
        assert len(result) == 10
        
        long_triple = {
            "subject": "a" * 60,
            "predicate": "b" * 40, 
            "object": "c" * 60
        }
        result = _validate_triples([long_triple])
        assert len(result[0]["subject"]) == 50
        assert len(result[0]["predicate"]) == 30
        assert len(result[0]["object"]) == 50
    
    def test_validate_triples_invalid(self):
        """Test invalid triples input."""
        assert _validate_triples("not_a_list") == []
        assert _validate_triples(None) == []
        assert _validate_triples([{"subject": "missing predicate"}]) == []
        assert _validate_triples([{"subject": "", "predicate": "test", "object": "obj"}]) == []


@pytest.mark.unit 
class TestEnrichmentHelpers:
    """Test helper functions."""
    
    def test_generate_default_title_short_content(self):
        """Test title generation for short content."""
        content = "This is a test"
        title = _generate_default_title(content)
        assert title == "This is a test"
    
    def test_generate_default_title_long_sentence(self):
        """Test title generation for long first sentence."""
        content = "This is a very long sentence that exceeds fifty characters and should be truncated properly."
        title = _generate_default_title(content)
        assert len(title) <= 50
        assert title.endswith("...")
    
    def test_generate_default_title_multiple_sentences(self):
        """Test title generation takes first sentence only."""
        content = "First sentence. Second sentence. Third sentence."
        title = _generate_default_title(content)
        assert title == "First sentence"
    
    def test_generate_default_title_empty(self):
        """Test title generation for empty content."""
        assert _generate_default_title("") == ""
        assert _generate_default_title("   ") == ""
    
    def test_build_enrichment_prompt(self):
        """Test enrichment prompt building."""
        content = "Test memory content"
        prompt = _build_enrichment_prompt(content)
        assert content in prompt
        assert "JSON format" in prompt
        assert "type" in prompt
        assert "weight" in prompt
        assert "triples" in prompt
    
    def test_default_enrichment(self):
        """Test fallback enrichment."""
        content = "Test content for default enrichment"
        result = _default_enrichment(content)
        
        assert result["type"] == "fact"
        assert result["weight"] == 0.5
        assert result["title"] == "Test content for default enrichment"
        assert result["summary"] == content
        assert result["tags"] == []
        assert result["validity_hours"] is None
        assert result["triples"] == []


@pytest.mark.unit
class TestEnrichmentParsing:
    """Test enrichment response parsing."""
    
    def test_parse_enrichment_response_valid_json(self):
        """Test parsing valid JSON response."""
        response_text = '''
        {
          "type": "fact",
          "weight": 0.8,
          "title": "API Test",
          "summary": "Testing the API endpoint",
          "tags": ["api", "test"],
          "validity_hours": 24,
          "triples": [{"subject": "API", "predicate": "has", "object": "test"}]
        }
        '''
        
        result = _parse_enrichment_response(response_text, "original content")
        
        assert result["type"] == "fact"
        assert result["weight"] == 0.8
        assert result["title"] == "API Test"
        assert result["tags"] == ["api", "test"]
        assert result["validity_hours"] == 24
        assert len(result["triples"]) == 1
    
    def test_parse_enrichment_response_with_extra_text(self):
        """Test parsing JSON with surrounding text."""
        response_text = '''
        Here's the analysis:
        {"type": "decision", "weight": 0.9, "title": "Important Decision", "summary": "A key decision", "tags": ["decision"], "validity_hours": null, "triples": []}
        
        End of analysis.
        '''
        
        result = _parse_enrichment_response(response_text, "original")
        assert result["type"] == "decision"
        assert result["weight"] == 0.9
        assert result["validity_hours"] is None
    
    def test_parse_enrichment_response_invalid_json(self):
        """Test parsing invalid JSON falls back to default."""
        response_text = "This is not valid JSON at all!"
        
        result = _parse_enrichment_response(response_text, "test content")
        
        # Should get default enrichment
        assert result["type"] == "fact"
        assert result["weight"] == 0.5
        assert result["title"] == "test content"
        assert result["summary"] == "test content"
    
    def test_parse_enrichment_response_missing_fields(self):
        """Test parsing JSON with missing fields uses defaults."""
        response_text = '{"type": "insight"}'  # Only type specified
        
        result = _parse_enrichment_response(response_text, "test content")
        
        assert result["type"] == "insight"
        assert result["weight"] == 0.5  # default
        assert "test content" in result["title"]  # generated from content


@pytest.mark.unit
class TestValidityFunctions:
    """Test validity/expiration functions."""
    
    def test_compute_validity_end_none(self):
        """Test computing validity end for permanent memories."""
        result = compute_validity_end(None)
        assert result is None
    
    def test_compute_validity_end_hours(self):
        """Test computing validity end for timed memories."""
        result = compute_validity_end(24)
        assert result is not None
        
        # Should be ISO format
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        
        # Should be roughly 24 hours from now (within 1 minute tolerance)
        expected = now + timedelta(hours=24)
        diff = abs((parsed.replace(tzinfo=timezone.utc) - expected).total_seconds())
        assert diff < 60  # within 1 minute
    
    def test_is_memory_expired_none(self):
        """Test non-expiring memories are never expired."""
        assert is_memory_expired(None) is False
    
    def test_is_memory_expired_future(self):
        """Test memories with future expiry are not expired."""
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        assert is_memory_expired(future.isoformat()) is False
    
    def test_is_memory_expired_past(self):
        """Test memories with past expiry are expired."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        assert is_memory_expired(past.isoformat()) is True
    
    def test_is_memory_expired_invalid_format(self):
        """Test invalid timestamp format."""
        assert is_memory_expired("invalid-timestamp") is False
        assert is_memory_expired("") is False


@pytest.mark.unit
class TestEnrichmentAPI:
    """Test main enrichment API functions."""
    
    @patch('memclawz.enrichment.GOOGLE_API_KEY', 'fake-key')
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_enrich_with_gemini_success(self, mock_model_class, mock_configure):
        """Test successful Gemini enrichment."""
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_response = MagicMock()
        mock_response.text = '{"type": "fact", "weight": 0.7, "title": "Test", "summary": "Summary", "tags": ["test"], "validity_hours": null, "triples": []}'
        mock_model.generate_content.return_value = mock_response
        
        result = _enrich_with_gemini("test content")
        
        assert result["type"] == "fact"
        assert result["weight"] == 0.7
        assert result["title"] == "Test"
        mock_configure.assert_called_once_with(api_key='fake-key')
    
    @patch('memclawz.enrichment.GOOGLE_API_KEY', None)
    def test_enrich_with_gemini_no_key(self):
        """Test Gemini enrichment without API key."""
        with pytest.raises(ValueError, match="GOOGLE_API_KEY not configured"):
            _enrich_with_gemini("test content")
    
    @patch('memclawz.enrichment.OPENAI_API_KEY', 'fake-key')
    @patch('openai.OpenAI')
    def test_enrich_with_openai_success(self, mock_openai):
        """Test successful OpenAI enrichment."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"type": "decision", "weight": 0.9, "title": "OpenAI Test", "summary": "OpenAI summary", "tags": ["openai"], "validity_hours": 48, "triples": []}'
        mock_client.chat.completions.create.return_value = mock_response
        
        result = _enrich_with_openai("test content")
        
        assert result["type"] == "decision"
        assert result["weight"] == 0.9
        assert result["validity_hours"] == 48
        
        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o-mini"
        assert call_args.kwargs["temperature"] == 0.1
    
    @patch('memclawz.enrichment.OPENAI_API_KEY', None)
    def test_enrich_with_openai_no_key(self):
        """Test OpenAI enrichment without API key."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY not configured"):
            _enrich_with_openai("test content")
    
    @patch('memclawz.enrichment._enrich_with_gemini')
    @patch('memclawz.enrichment._enrich_with_openai')
    def test_enrich_memory_gemini_success(self, mock_openai, mock_gemini):
        """Test enrich_memory tries Gemini first."""
        expected_result = {"type": "fact", "weight": 0.8, "title": "Gemini", "summary": "From Gemini", "tags": [], "validity_hours": None, "triples": []}
        mock_gemini.return_value = expected_result
        
        result = enrich_memory("test content")
        
        assert result == expected_result
        mock_gemini.assert_called_once_with("test content")
        mock_openai.assert_not_called()
    
    @patch('memclawz.enrichment._enrich_with_gemini')
    @patch('memclawz.enrichment._enrich_with_openai')
    def test_enrich_memory_gemini_fail_openai_success(self, mock_openai, mock_gemini):
        """Test enrich_memory falls back to OpenAI when Gemini fails."""
        mock_gemini.side_effect = Exception("Gemini failed")
        expected_result = {"type": "fact", "weight": 0.8, "title": "OpenAI", "summary": "From OpenAI", "tags": [], "validity_hours": None, "triples": []}
        mock_openai.return_value = expected_result
        
        result = enrich_memory("test content")
        
        assert result == expected_result
        mock_gemini.assert_called_once()
        mock_openai.assert_called_once_with("test content")
    
    @patch('memclawz.enrichment._enrich_with_gemini')
    @patch('memclawz.enrichment._enrich_with_openai')
    @patch('memclawz.enrichment._default_enrichment')
    def test_enrich_memory_both_fail(self, mock_default, mock_openai, mock_gemini):
        """Test enrich_memory falls back to default when both APIs fail."""
        mock_gemini.side_effect = Exception("Gemini failed")
        mock_openai.side_effect = Exception("OpenAI failed")
        expected_result = {"type": "fact", "weight": 0.5, "title": "Default", "summary": "Default", "tags": [], "validity_hours": None, "triples": []}
        mock_default.return_value = expected_result
        
        result = enrich_memory("test content")
        
        assert result == expected_result
        mock_gemini.assert_called_once()
        mock_openai.assert_called_once()
        mock_default.assert_called_once_with("test content")


@pytest.mark.unit
class TestEnrichmentEdgeCases:
    """Test edge cases and error handling."""
    
    def test_enrich_empty_string(self):
        """Test enriching empty string."""
        with patch('memclawz.enrichment._enrich_with_gemini', side_effect=Exception("API failed")):
            with patch('memclawz.enrichment._enrich_with_openai', side_effect=Exception("API failed")):
                result = enrich_memory("")
                
                assert result["type"] == "fact"
                assert result["title"] == ""
                assert result["summary"] == ""
    
    def test_enrich_unicode_content(self):
        """Test enriching unicode content."""
        unicode_content = "Test with émojis 🚀 and ñiño characters"
        
        with patch('memclawz.enrichment._enrich_with_gemini', side_effect=Exception("API failed")):
            with patch('memclawz.enrichment._enrich_with_openai', side_effect=Exception("API failed")):
                result = enrich_memory(unicode_content)
                
                assert result["summary"] == unicode_content
                assert "émojis" in result["title"] or "Test with" in result["title"]
    
    def test_enrich_very_long_content(self):
        """Test enriching very long content."""
        long_content = "A" * 1000  # 1000 character string
        
        with patch('memclawz.enrichment._enrich_with_gemini', side_effect=Exception("API failed")):
            with patch('memclawz.enrichment._enrich_with_openai', side_effect=Exception("API failed")):
                result = enrich_memory(long_content)
                
                # Title should be truncated
                assert len(result["title"]) <= 100
                # Summary should be truncated  
                assert len(result["summary"]) <= 200
                assert result["summary"].startswith("A")
    
    def test_validity_computation_edge_cases(self):
        """Test validity computation edge cases."""
        # Zero hours should be handled
        result = compute_validity_end(0)
        # Should still generate a timestamp (validation happens elsewhere)
        assert result is not None
        
        # Very large hours
        result = compute_validity_end(10000)
        assert result is not None
        
        # Negative hours 
        result = compute_validity_end(-5)
        assert result is not None