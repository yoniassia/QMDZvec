"""MemClawz v6 — Sleep-Time Reflection Engine.

Analyzes recent memories to detect patterns, contradictions,
generate insights, and propose MEMORY.md updates.
"""
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import anthropic
from mem0 import Memory

from .config import MEM0_CONFIG, ANTHROPIC_API_KEY, STATE_DIR, WORKSPACE_DIR
from .utils import load_json, save_json, utcnow_iso

logger = logging.getLogger(__name__)

REFLECTION_STATE_FILE = STATE_DIR / "reflection_state.json"
MEMORY_DIR = WORKSPACE_DIR / "memory"
MEMORY_MD = WORKSPACE_DIR / "MEMORY.md"


class ReflectionEngine:
    """Analyzes recent memories for patterns, contradictions, and insights."""

    def __init__(self, mem: Memory | None = None):
        self.mem = mem or Memory.from_config(MEM0_CONFIG)
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def reflect(self, hours: int = 24, max_memories: int = 100) -> dict:
        """Run a reflection cycle.

        1. Gather recent memories (last N hours or since last reflection)
        2. Analyze for patterns, contradictions, insights
        3. Write results to reflection file
        4. Return proposed actions

        Args:
            hours: How far back to look (default 24h).
            max_memories: Maximum memories to analyze.

        Returns:
            Reflection results with insights and proposed actions.
        """
        state = load_json(REFLECTION_STATE_FILE)
        last_reflection = state.get("last_reflection")

        # Determine cutoff
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        if last_reflection:
            try:
                last_dt = datetime.fromisoformat(last_reflection.replace("Z", "+00:00"))
                # Use whichever is more recent
                cutoff = max(cutoff, last_dt)
            except ValueError:
                pass

        # Gather recent memories
        all_mems = self.mem.get_all(user_id="yoni", limit=10000)
        if isinstance(all_mems, dict):
            all_mems = all_mems.get("results", [])

        recent = []
        for m in all_mems:
            meta = m.get("metadata", {})
            extracted = meta.get("extracted_at", meta.get("created_at", ""))
            if extracted:
                try:
                    ext_dt = datetime.fromisoformat(extracted.replace("Z", "+00:00"))
                    if ext_dt >= cutoff:
                        recent.append(m)
                except ValueError:
                    pass

        recent = recent[:max_memories]

        if not recent:
            return {
                "status": "empty",
                "message": "No recent memories to reflect on",
                "since": cutoff.isoformat(),
            }

        # Build context for reflection
        memory_texts = []
        for m in recent:
            meta = m.get("metadata", {})
            text = m.get("memory", m.get("content", ""))
            memory_texts.append(
                f"[{meta.get('type', 'fact')}] [{meta.get('agent', 'unknown')}] {text}"
            )

        context = "\n".join(memory_texts)

        # Run LLM reflection
        reflection = self._reflect_with_llm(context)

        # Write reflection file
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
        reflection_file = MEMORY_DIR / f"reflection-{now_str}.md"

        md_content = self._build_reflection_markdown(reflection, recent)
        reflection_file.write_text(md_content)

        # Store insights as memories
        insights_added = 0
        for insight in reflection.get("insights", []):
            if len(insight) > 20:
                self.mem.add(
                    insight,
                    user_id="yoni",
                    metadata={
                        "source": "reflection",
                        "type": "insight",
                        "agent": "system",
                        "extracted_at": utcnow_iso(),
                        "reflection": True,
                    },
                )
                insights_added += 1

        # Update state
        state["last_reflection"] = utcnow_iso()
        state["reflections_total"] = state.get("reflections_total", 0) + 1
        state["last_insights_count"] = len(reflection.get("insights", []))
        state["last_contradictions_count"] = len(reflection.get("contradictions", []))
        save_json(state, REFLECTION_STATE_FILE)

        return {
            "status": "reflected",
            "memories_analyzed": len(recent),
            "insights": reflection.get("insights", []),
            "contradictions": reflection.get("contradictions", []),
            "patterns": reflection.get("patterns", []),
            "proposed_memory_updates": reflection.get("proposed_updates", []),
            "proposed_archives": reflection.get("proposed_archives", []),
            "insights_stored": insights_added,
            "file": str(reflection_file),
        }

    def _reflect_with_llm(self, memory_context: str) -> dict:
        """Use LLM to analyze memories and generate reflection."""
        prompt = f"""You are a memory reflection engine for an AI agent fleet.
Analyze these recent memories and provide structured insights.

Memories:
{memory_context[:12000]}

Analyze and return a JSON object with:
- "insights": list of high-level patterns or lessons learned (strings)
- "contradictions": list of {{old: "...", new: "...", resolution: "..."}} objects where memories conflict
- "patterns": list of recurring themes or behaviors noticed
- "proposed_updates": list of {{fact: "...", action: "update|add|remove", reason: "..."}} for MEMORY.md
- "proposed_archives": list of memory IDs that seem stale/redundant (strings)
- "summary": 2-3 sentence reflection summary

Return ONLY valid JSON."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # Strip markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        if text.startswith("json"):
            text = text[4:]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            logger.warning("Reflection LLM returned invalid JSON")
            return {
                "insights": [],
                "contradictions": [],
                "patterns": [],
                "proposed_updates": [],
                "proposed_archives": [],
                "summary": text[:500],
            }

    def _build_reflection_markdown(self, reflection: dict, memories: list) -> str:
        lines = [
            f"# Reflection — {utcnow_iso()[:10]}",
            f"**Generated:** {utcnow_iso()}",
            f"**Memories analyzed:** {len(memories)}",
            "",
            "## Summary",
            reflection.get("summary", "No summary."),
            "",
        ]

        insights = reflection.get("insights", [])
        if insights:
            lines.append("## Insights")
            for i in insights:
                lines.append(f"- {i}")
            lines.append("")

        patterns = reflection.get("patterns", [])
        if patterns:
            lines.append("## Patterns")
            for p in patterns:
                lines.append(f"- {p}")
            lines.append("")

        contradictions = reflection.get("contradictions", [])
        if contradictions:
            lines.append("## Contradictions Found")
            for c in contradictions:
                if isinstance(c, dict):
                    lines.append(f"- **Old:** {c.get('old', '?')}")
                    lines.append(f"  **New:** {c.get('new', '?')}")
                    lines.append(f"  **Resolution:** {c.get('resolution', '?')}")
                else:
                    lines.append(f"- {c}")
            lines.append("")

        updates = reflection.get("proposed_updates", [])
        if updates:
            lines.append("## Proposed MEMORY.md Updates")
            for u in updates:
                if isinstance(u, dict):
                    lines.append(f"- [{u.get('action', '?')}] {u.get('fact', '?')} — {u.get('reason', '')}")
                else:
                    lines.append(f"- {u}")
            lines.append("")

        return "\n".join(lines)
