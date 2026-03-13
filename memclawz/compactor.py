"""MemClawz v6 — Compaction Engine.

Three-tier compaction:
  1. Session Compactor — triggered when a session exceeds context threshold
  2. Daily Digest — runs at midnight UTC, cross-references the day's memories
  3. Weekly Merge — runs Sunday midnight, merges into MEMORY.md, deduplicates
"""
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import anthropic
from mem0 import Memory

from .config import (
    MEM0_CONFIG,
    ANTHROPIC_API_KEY,
    STATE_DIR,
    WORKSPACE_DIR,
)
from .classifier import classify_memory
from .scoring import composite_score
from .utils import load_json, save_json, utcnow_iso

logger = logging.getLogger(__name__)

COMPACTION_STATE_FILE = STATE_DIR / "compaction_state.json"
SESSIONS_DIR = WORKSPACE_DIR / "memory" / "sessions"
MEMORY_DIR = WORKSPACE_DIR / "memory"
ARCHIVE_DIR = MEMORY_DIR / "archive"


def _get_llm_client():
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _extract_facts_with_llm(content: str, context: str = "") -> dict[str, Any]:
    """Use LLM to extract structured facts from session content.

    Returns:
        Dict with keys: facts, decisions, actions, people, topics, pending
    """
    client = _get_llm_client()
    prompt = f"""Extract structured information from this conversation content.
{f"Context: {context}" if context else ""}

Content:
{content[:8000]}

Return a JSON object with these keys:
- "facts": list of factual statements learned
- "decisions": list of decisions made
- "actions": list of actions taken or committed to
- "people": list of people mentioned with their roles/context
- "topics": list of main topics discussed
- "pending": list of unresolved items or follow-ups
- "summary": 2-3 sentence summary of the session

Return ONLY valid JSON, no markdown formatting."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    if text.startswith("json"):
        text = text[4:]

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        logger.warning(f"LLM returned invalid JSON, using raw text as summary")
        return {
            "facts": [],
            "decisions": [],
            "actions": [],
            "people": [],
            "topics": [],
            "pending": [],
            "summary": text[:500],
        }


class SessionCompactor:
    """Compacts a single session's context into structured memories."""

    def __init__(self, mem: Memory | None = None):
        self.mem = mem or Memory.from_config(MEM0_CONFIG)

    def compact_session(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        agent_id: str = "main",
        force: bool = False,
    ) -> dict[str, Any]:
        """Compact a session's messages into structured memories.

        Args:
            session_id: Unique session identifier.
            messages: List of message dicts with 'role' and 'content'.
            agent_id: Which agent owns this session.
            force: Force compaction even if below threshold.

        Returns:
            Compaction result with summary and extracted memories.
        """
        if not messages:
            return {"status": "empty", "session_id": session_id}

        # Build content from messages
        content = "\n".join(
            f"[{m.get('role', 'unknown')}]: {m.get('content', '')[:500]}"
            for m in messages[-50:]  # Last 50 messages max
        )

        # Extract facts
        extraction = _extract_facts_with_llm(
            content,
            context=f"Session: {session_id}, Agent: {agent_id}",
        )

        # Write to canonical session file
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        slug = session_id.replace("/", "_").replace(":", "_")
        session_file = SESSIONS_DIR / f"{slug}.md"

        md_content = self._build_session_markdown(session_id, agent_id, extraction)
        session_file.write_text(md_content)
        logger.info(f"Session compaction written to {session_file}")

        # Feed extracted facts to Mem0/Qdrant
        memories_added = 0
        for fact in extraction.get("facts", []):
            if len(fact) > 20:
                mem_type = classify_memory(fact)
                self.mem.add(
                    fact,
                    user_id="yoni",
                    metadata={
                        "source": f"compaction:session:{session_id}",
                        "type": mem_type,
                        "agent": agent_id,
                        "session_id": session_id,
                        "extracted_at": utcnow_iso(),
                        "compacted": True,
                    },
                )
                memories_added += 1

        for decision in extraction.get("decisions", []):
            if len(decision) > 20:
                self.mem.add(
                    decision,
                    user_id="yoni",
                    metadata={
                        "source": f"compaction:session:{session_id}",
                        "type": "decision",
                        "agent": agent_id,
                        "session_id": session_id,
                        "extracted_at": utcnow_iso(),
                        "compacted": True,
                    },
                )
                memories_added += 1

        # Update compaction state
        state = load_json(COMPACTION_STATE_FILE)
        sessions = state.get("sessions", {})
        sessions[session_id] = {
            "last_compacted": utcnow_iso(),
            "messages_compacted": len(messages),
            "memories_added": memories_added,
            "agent": agent_id,
        }
        state["sessions"] = sessions
        state["last_compaction"] = utcnow_iso()
        save_json(state, COMPACTION_STATE_FILE)

        return {
            "status": "compacted",
            "session_id": session_id,
            "file": str(session_file),
            "memories_added": memories_added,
            "summary": extraction.get("summary", ""),
            "facts_count": len(extraction.get("facts", [])),
            "decisions_count": len(extraction.get("decisions", [])),
            "pending": extraction.get("pending", []),
        }

    def _build_session_markdown(
        self, session_id: str, agent_id: str, extraction: dict
    ) -> str:
        """Build a markdown summary of a compacted session."""
        lines = [
            f"# Session: {session_id}",
            f"**Agent:** {agent_id}",
            f"**Compacted:** {utcnow_iso()}",
            "",
            "## Summary",
            extraction.get("summary", "No summary available."),
            "",
        ]
        for section, key in [
            ("Facts", "facts"),
            ("Decisions", "decisions"),
            ("Actions", "actions"),
            ("People", "people"),
            ("Pending Items", "pending"),
        ]:
            items = extraction.get(key, [])
            if items:
                lines.append(f"## {section}")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")

        topics = extraction.get("topics", [])
        if topics:
            lines.append(f"## Topics")
            lines.append(", ".join(topics))
            lines.append("")

        return "\n".join(lines)


class DailyDigest:
    """Creates a daily digest from all sessions active that day."""

    def __init__(self, mem: Memory | None = None):
        self.mem = mem or Memory.from_config(MEM0_CONFIG)

    def generate(self, date: str | None = None) -> dict[str, Any]:
        """Generate a daily digest.

        Args:
            date: Date in YYYY-MM-DD format (defaults to today).

        Returns:
            Digest result with file path and summary.
        """
        date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Gather session files updated today
        session_files = []
        if SESSIONS_DIR.exists():
            for f in SESSIONS_DIR.glob("*.md"):
                try:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                    if mtime.strftime("%Y-%m-%d") == date:
                        session_files.append(f)
                except OSError:
                    continue

        # Gather memories from today
        all_mems = self.mem.get_all(user_id="yoni", limit=10000)
        if isinstance(all_mems, dict):
            all_mems = all_mems.get("results", [])

        today_mems = []
        for m in all_mems:
            meta = m.get("metadata", {})
            extracted = meta.get("extracted_at", "")
            if extracted.startswith(date):
                today_mems.append(m)

        if not session_files and not today_mems:
            return {"status": "empty", "date": date, "message": "No activity for this date"}

        # Build digest content
        content_parts = []
        for sf in session_files:
            content_parts.append(sf.read_text()[:2000])
        for m in today_mems[:50]:
            content_parts.append(m.get("memory", m.get("content", "")))

        combined = "\n---\n".join(content_parts)

        # Use LLM to create digest
        extraction = _extract_facts_with_llm(
            combined,
            context=f"Daily digest for {date}",
        )

        # Deduplicate facts against existing memories
        deduped_facts = self._deduplicate_facts(extraction.get("facts", []))

        # Write daily digest file
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        digest_file = MEMORY_DIR / f"{date}.md"
        digest_content = self._build_digest_markdown(date, extraction, deduped_facts)
        digest_file.write_text(digest_content)

        # Add new unique facts to memory
        added = 0
        for fact in deduped_facts:
            if len(fact) > 20:
                mem_type = classify_memory(fact)
                self.mem.add(
                    fact,
                    user_id="yoni",
                    metadata={
                        "source": f"daily-digest:{date}",
                        "type": mem_type,
                        "agent": "system",
                        "extracted_at": utcnow_iso(),
                        "digest": True,
                    },
                )
                added += 1

        return {
            "status": "generated",
            "date": date,
            "file": str(digest_file),
            "sessions_processed": len(session_files),
            "memories_today": len(today_mems),
            "new_facts": len(deduped_facts),
            "memories_added": added,
            "summary": extraction.get("summary", ""),
        }

    def _deduplicate_facts(self, facts: list[str], threshold: float = 0.88) -> list[str]:
        """Remove facts that are too similar to existing memories."""
        unique = []
        for fact in facts:
            if len(fact) < 20:
                continue
            try:
                similar = self.mem.search(fact, user_id="yoni", limit=3)
                if isinstance(similar, dict):
                    similar = similar.get("results", [])
                is_dup = any(r.get("score", 0) > threshold for r in similar)
                if not is_dup:
                    unique.append(fact)
            except Exception:
                unique.append(fact)
        return unique

    def _build_digest_markdown(
        self, date: str, extraction: dict, deduped_facts: list[str]
    ) -> str:
        lines = [
            f"# Daily Digest — {date}",
            f"**Generated:** {utcnow_iso()}",
            "",
            "## Summary",
            extraction.get("summary", "No summary."),
            "",
        ]
        if deduped_facts:
            lines.append("## New Facts")
            for f in deduped_facts:
                lines.append(f"- {f}")
            lines.append("")

        for section, key in [
            ("Decisions", "decisions"),
            ("Pending Items", "pending"),
        ]:
            items = extraction.get(key, [])
            if items:
                lines.append(f"## {section}")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")

        return "\n".join(lines)


class WeeklyMerge:
    """Weekly merge: consolidates daily digests, updates MEMORY.md, archives old data."""

    def __init__(self, mem: Memory | None = None):
        self.mem = mem or Memory.from_config(MEM0_CONFIG)

    def merge(self) -> dict[str, Any]:
        """Run the weekly merge process.

        1. Read all daily digests from the past week
        2. Merge into MEMORY.md updates
        3. Archive stale daily logs (>14 days)
        4. Run Qdrant deduplication
        5. Report memory health stats
        """
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        two_weeks_ago = now - timedelta(days=14)

        # 1. Gather weekly digests
        weekly_content = []
        digest_files = []
        if MEMORY_DIR.exists():
            for f in sorted(MEMORY_DIR.glob("????-??-??.md")):
                try:
                    file_date = datetime.strptime(f.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    if file_date >= week_ago:
                        weekly_content.append(f.read_text()[:3000])
                        digest_files.append(f.name)
                except ValueError:
                    continue

        # 2. Generate weekly summary
        if weekly_content:
            combined = "\n---\n".join(weekly_content)
            extraction = _extract_facts_with_llm(
                combined,
                context=f"Weekly merge for week ending {now.strftime('%Y-%m-%d')}",
            )
        else:
            extraction = {"summary": "No activity this week.", "facts": [], "decisions": []}

        # 3. Archive stale files
        archived = []
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        if MEMORY_DIR.exists():
            for f in MEMORY_DIR.glob("????-??-??.md"):
                try:
                    file_date = datetime.strptime(f.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    if file_date < two_weeks_ago:
                        dest = ARCHIVE_DIR / f.name
                        f.rename(dest)
                        archived.append(f.name)
                except (ValueError, OSError):
                    continue

        # 4. Qdrant deduplication
        dedup_stats = self._deduplicate_qdrant()

        # 5. Write weekly report
        report_file = MEMORY_DIR / f"weekly-{now.strftime('%Y-%m-%d')}.md"
        report = self._build_weekly_report(
            now, extraction, digest_files, archived, dedup_stats
        )
        report_file.write_text(report)

        # Update compaction state
        state = load_json(COMPACTION_STATE_FILE)
        state["last_weekly_merge"] = utcnow_iso()
        state["weekly_merges"] = state.get("weekly_merges", 0) + 1
        save_json(state, COMPACTION_STATE_FILE)

        return {
            "status": "merged",
            "digests_processed": len(digest_files),
            "files_archived": len(archived),
            "dedup_removed": dedup_stats.get("removed", 0),
            "report_file": str(report_file),
            "summary": extraction.get("summary", ""),
        }

    def _deduplicate_qdrant(self, threshold: float = 0.92) -> dict:
        """Find and remove near-duplicate vectors in Qdrant."""
        try:
            all_mems = self.mem.get_all(user_id="yoni", limit=10000)
            if isinstance(all_mems, dict):
                all_mems = all_mems.get("results", [])

            # Sample-based dedup: check random memories for duplicates
            removed = 0
            checked = 0
            seen_ids = set()

            for m in all_mems[:500]:  # Check up to 500 memories
                mid = m.get("id")
                if not mid or mid in seen_ids:
                    continue
                seen_ids.add(mid)
                checked += 1

                content = m.get("memory", "")
                if not content:
                    continue

                # Search for similar
                similar = self.mem.search(content, user_id="yoni", limit=5)
                if isinstance(similar, dict):
                    similar = similar.get("results", [])

                for s in similar:
                    sid = s.get("id")
                    if sid and sid != mid and sid not in seen_ids and s.get("score", 0) > threshold:
                        # Keep the one with more metadata / newer timestamp
                        try:
                            self.mem.delete(sid)
                            seen_ids.add(sid)
                            removed += 1
                        except Exception:
                            pass

            return {"checked": checked, "removed": removed}
        except Exception as e:
            logger.error(f"Deduplication error: {e}")
            return {"checked": 0, "removed": 0, "error": str(e)}

    def _build_weekly_report(
        self, now, extraction, digests, archived, dedup_stats
    ) -> str:
        lines = [
            f"# Weekly Merge Report — {now.strftime('%Y-%m-%d')}",
            f"**Generated:** {utcnow_iso()}",
            "",
            "## Summary",
            extraction.get("summary", "No summary."),
            "",
            "## Stats",
            f"- Daily digests processed: {len(digests)}",
            f"- Files archived: {len(archived)}",
            f"- Duplicates removed: {dedup_stats.get('removed', 0)}",
            f"- Memories checked: {dedup_stats.get('checked', 0)}",
            "",
        ]

        key_facts = extraction.get("facts", [])
        if key_facts:
            lines.append("## Key Facts This Week")
            for f in key_facts:
                lines.append(f"- {f}")
            lines.append("")

        decisions = extraction.get("decisions", [])
        if decisions:
            lines.append("## Decisions This Week")
            for d in decisions:
                lines.append(f"- {d}")
            lines.append("")

        if archived:
            lines.append("## Archived Files")
            for a in archived:
                lines.append(f"- {a}")
            lines.append("")

        return "\n".join(lines)


def get_compaction_status() -> dict:
    """Get current compaction status and health metrics."""
    state = load_json(COMPACTION_STATE_FILE)

    # Count session files
    session_count = 0
    if SESSIONS_DIR.exists():
        session_count = len(list(SESSIONS_DIR.glob("*.md")))

    # Count daily digests
    digest_count = 0
    if MEMORY_DIR.exists():
        digest_count = len(list(MEMORY_DIR.glob("????-??-??.md")))

    # Count archived
    archive_count = 0
    if ARCHIVE_DIR.exists():
        archive_count = len(list(ARCHIVE_DIR.glob("*.md")))

    return {
        "status": "ok",
        "last_compaction": state.get("last_compaction"),
        "last_weekly_merge": state.get("last_weekly_merge"),
        "weekly_merges_total": state.get("weekly_merges", 0),
        "session_files": session_count,
        "daily_digests": digest_count,
        "archived_files": archive_count,
        "sessions_compacted": len(state.get("sessions", {})),
    }
