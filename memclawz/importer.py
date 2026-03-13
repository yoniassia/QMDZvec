"""Bulk import utilities for MemClawz — import from markdown, SQLite, JSONL."""
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from mem0 import Memory

from .config import MEM0_CONFIG
from .classifier import classify_memory


def import_markdown(filepath: str, mem: Memory | None = None, user_id: str = "yoni", agent: str = "main") -> int:
    """Import memories from a markdown file (one paragraph = one memory)."""
    mem = mem or Memory.from_config(MEM0_CONFIG)
    text = Path(filepath).read_text()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 20]

    count = 0
    for para in paragraphs:
        mem_type = classify_memory(para)
        mem.add(
            para,
            user_id=user_id,
            metadata={
                "source": f"import-markdown:{Path(filepath).name}",
                "type": mem_type,
                "agent": agent,
                "extracted_at": datetime.utcnow().isoformat(),
                "auto_extracted": True,
            },
        )
        count += 1
    return count


def import_jsonl(filepath: str, mem: Memory | None = None, user_id: str = "yoni") -> int:
    """Import memories from a JSONL file. Each line: {"content": "...", "metadata": {...}}."""
    mem = mem or Memory.from_config(MEM0_CONFIG)
    count = 0
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            content = data.get("content", "")
            if not content or len(content) < 20:
                continue
            metadata = data.get("metadata", {})
            if "type" not in metadata:
                metadata["type"] = classify_memory(content)
            metadata.setdefault("source", f"import-jsonl:{Path(filepath).name}")
            metadata.setdefault("extracted_at", datetime.utcnow().isoformat())
            mem.add(content, user_id=user_id, metadata=metadata)
            count += 1
    return count


def import_sqlite(
    db_path: str,
    table: str = "summaries",
    content_col: str = "content",
    mem: Memory | None = None,
    user_id: str = "yoni",
) -> int:
    """Import from a SQLite table."""
    mem = mem or Memory.from_config(MEM0_CONFIG)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    conn.close()

    count = 0
    for row in rows:
        content = dict(row).get(content_col, "")
        if not content or len(content) < 20:
            continue
        mem_type = classify_memory(content)
        mem.add(
            content,
            user_id=user_id,
            metadata={
                "source": f"import-sqlite:{Path(db_path).name}:{table}",
                "type": mem_type,
                "extracted_at": datetime.utcnow().isoformat(),
                "auto_extracted": True,
            },
        )
        count += 1
    return count
