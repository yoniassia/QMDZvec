"""Tests for LCM watcher (unit tests, no live DB required)."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from memclawz.utils import load_json, save_json


def test_load_json_missing_file():
    assert load_json("/tmp/nonexistent_memclawz_test.json") == {}


def test_save_and_load_json():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    data = {"last_summary_id": 42, "total_extracted": 10}
    save_json(data, path)
    loaded = load_json(path)
    assert loaded["last_summary_id"] == 42
    assert loaded["total_extracted"] == 10
    Path(path).unlink()


def test_extract_agent_from_session_fallback():
    from memclawz.utils import extract_agent_from_session
    assert extract_agent_from_session("nonexistent", "/tmp/no_such_dir") == "main"
