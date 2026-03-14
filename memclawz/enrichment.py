"""MemClawz v7 — Auto-Enrichment Layer.

Phase 1: Auto-Enrichment
Phase 2: Temporal Validity
Phase 3: RDF-Lite Triples
"""
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def enrich_memory(content: str) -> Dict[str, Any]:
    """Enrich raw memory content using Google Gemini or OpenAI as fallback.
    
    Args:
        content: Raw plain text from agent
        
    Returns:
        Dict with type, weight, title, summary, tags, validity_hours, triples
    """
    try:
        return _enrich_with_gemini(content)
    except Exception as e:
        logger.warning(f"Gemini enrichment failed: {e}, falling back to OpenAI")
        try:
            return _enrich_with_openai(content)
        except Exception as e2:
            logger.warning(f"OpenAI enrichment also failed: {e2}, using defaults")
            return _default_enrichment(content)


def _enrich_with_gemini(content: str) -> Dict[str, Any]:
    """Enrich using Google Gemini 2.5 Flash-Lite."""
    try:
        import google.generativeai as genai
        from .config import GOOGLE_API_KEY, GEMINI_MODEL
        
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not configured")
            
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = _build_enrichment_prompt(content)
        response = model.generate_content(prompt)
        
        return _parse_enrichment_response(response.text, content)
        
    except ImportError:
        raise ValueError("google-generativeai package not installed")
    except Exception as e:
        raise ValueError(f"Gemini API error: {e}")


def _enrich_with_openai(content: str) -> Dict[str, Any]:
    """Fallback enrichment using OpenAI."""
    from openai import OpenAI
    from .config import OPENAI_API_KEY
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    prompt = _build_enrichment_prompt(content)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a memory enrichment assistant. Follow the format exactly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=500
    )
    
    return _parse_enrichment_response(response.choices[0].message.content, content)


def _build_enrichment_prompt(content: str) -> str:
    """Build the enrichment prompt for LLM."""
    return f"""Analyze this memory content and provide enrichment metadata:

CONTENT: {content}

Provide EXACTLY this JSON format (no additional text):
{{
  "type": "<decision|fact|event|insight|preference|relationship|procedure>",
  "weight": <float 0.0-1.0>,
  "title": "<brief descriptive title>",
  "summary": "<1-2 sentence summary>",
  "tags": ["<tag1>", "<tag2>", "<tag3>"],
  "validity_hours": <hours as integer, or null for permanent>,
  "triples": [
    {{"subject": "<entity1>", "predicate": "<relationship>", "object": "<entity2>"}},
    {{"subject": "<entity2>", "predicate": "<property>", "object": "<value>"}}
  ]
}}

GUIDELINES:
- type: choose most fitting category
- weight: 0.8-1.0 for important/decision, 0.5-0.7 for facts, 0.3-0.5 for transient
- validity_hours: 1-24 for prices/weather, 168-720 for semi-stable, null for permanent
- tags: 3-5 relevant keywords
- triples: extract key subject-predicate-object relationships from the content"""


def _parse_enrichment_response(response_text: str, original_content: str) -> Dict[str, Any]:
    """Parse LLM response into enrichment dict."""
    import json
    
    try:
        # Extract JSON from response (in case there's extra text)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
            
        data = json.loads(response_text)
        
        # Validate and clean the response
        result = {
            "type": _validate_type(data.get("type", "fact")),
            "weight": _validate_weight(data.get("weight", 0.5)),
            "title": data.get("title", _generate_default_title(original_content))[:100],
            "summary": data.get("summary", original_content[:200]),
            "tags": _validate_tags(data.get("tags", [])),
            "validity_hours": _validate_validity_hours(data.get("validity_hours")),
            "triples": _validate_triples(data.get("triples", []))
        }
        
        return result
        
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        logger.warning(f"Failed to parse enrichment response: {e}")
        return _default_enrichment(original_content)


def _validate_type(type_str: str) -> str:
    """Validate memory type."""
    valid_types = {"decision", "fact", "event", "insight", "preference", "relationship", "procedure"}
    return type_str if type_str in valid_types else "fact"


def _validate_weight(weight: Any) -> float:
    """Validate weight is between 0 and 1."""
    try:
        w = float(weight)
        return max(0.0, min(1.0, w))
    except (ValueError, TypeError):
        return 0.5


def _validate_tags(tags: Any) -> List[str]:
    """Validate tags list."""
    if not isinstance(tags, list):
        return []
    return [str(tag)[:20] for tag in tags if tag][:5]  # Max 5 tags, 20 chars each


def _validate_validity_hours(validity: Any) -> Optional[int]:
    """Validate validity_hours."""
    if validity is None:
        return None
    try:
        hours = int(validity)
        return max(1, min(8760, hours))  # 1 hour to 1 year max
    except (ValueError, TypeError):
        return None


def _validate_triples(triples: Any) -> List[Dict[str, str]]:
    """Validate triples list."""
    if not isinstance(triples, list):
        return []
    
    validated = []
    for triple in triples[:10]:  # Max 10 triples
        if isinstance(triple, dict):
            subject = str(triple.get("subject", ""))[:50]
            predicate = str(triple.get("predicate", ""))[:30]
            obj = str(triple.get("object", ""))[:50]
            
            if subject and predicate and obj:
                validated.append({
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj
                })
    
    return validated


def _generate_default_title(content: str) -> str:
    """Generate a default title from content."""
    # Take first sentence or first 50 chars
    first_sentence = content.split('.')[0].strip()
    if len(first_sentence) > 50:
        return first_sentence[:47] + "..."
    return first_sentence or content[:50]


def _default_enrichment(content: str) -> Dict[str, Any]:
    """Fallback enrichment when LLM fails."""
    return {
        "type": "fact",
        "weight": 0.5,
        "title": _generate_default_title(content),
        "summary": content[:200],
        "tags": [],
        "validity_hours": None,
        "triples": []
    }


def compute_validity_end(validity_hours: Optional[int]) -> Optional[str]:
    """Compute ts_valid_end from validity_hours."""
    if validity_hours is None:
        return None
    
    end_time = datetime.now(timezone.utc) + timedelta(hours=validity_hours)
    return end_time.isoformat()


def is_memory_expired(ts_valid_end: Optional[str]) -> bool:
    """Check if memory has expired based on ts_valid_end."""
    if ts_valid_end is None:
        return False
    
    try:
        end_time = datetime.fromisoformat(ts_valid_end.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) > end_time.replace(tzinfo=timezone.utc if end_time.tzinfo is None else end_time.tzinfo)
    except (ValueError, TypeError):
        return False