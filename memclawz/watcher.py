"""LCM → Mem0 → Qdrant (+ Graphiti v6) auto-extract pipeline."""
import asyncio
import sqlite3
import time
import sys
from datetime import datetime
from mem0 import Memory

from .config import MEM0_CONFIG, LCM_DB_PATH, AGENTS_DIR, STATE_DIR, SYNC_INTERVAL, GRAPHITI_ENABLED
from .classifier import classify_memory
from .utils import load_json, save_json, extract_agent_from_session

STATE_FILE = STATE_DIR / "last_sync.json"


class LCMWatcher:
    def __init__(self):
        self.mem = Memory.from_config(MEM0_CONFIG)
        self.db_path = LCM_DB_PATH

    def get_new_summaries(self, since_id: int = 0) -> list[dict]:
        """Fetch summaries from LCM database newer than since_id."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT rowid, * FROM summaries WHERE rowid > ? ORDER BY created_at",
            (since_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def sync(self) -> int:
        """Run one sync cycle. Returns number of memories extracted."""
        state = load_json(STATE_FILE)
        last_id = state.get("last_summary_id", 0)
        summaries = self.get_new_summaries(last_id)

        imported = 0
        graphiti_fed = 0
        for s in summaries:
            content = s.get("content", "")
            if not content or len(content) < 20:
                last_id = max(last_id, s.get("rowid", 0))
                continue

            conv_id = s.get("conversation_id", "")
            mem_type = classify_memory(content)
            agent = extract_agent_from_session(str(conv_id), AGENTS_DIR) if conv_id else "main"

            try:
                self.mem.add(
                    content,
                    user_id="yoni",
                    metadata={
                        "source": "lcm-auto-extract",
                        "type": mem_type,
                        "agent": agent,
                        "conversation_id": str(conv_id),
                        "extracted_at": datetime.utcnow().isoformat(),
                        "auto_extracted": True,
                    },
                )
                imported += 1

                # Feed to Graphiti (v6)
                if GRAPHITI_ENABLED:
                    try:
                        from .graphiti_layer import add_episode
                        asyncio.get_event_loop().run_until_complete(
                            add_episode(
                                content=content,
                                agent_id=agent,
                                source=f"lcm:{conv_id}",
                            )
                        )
                        graphiti_fed += 1
                    except RuntimeError:
                        # No event loop running, create one
                        loop = asyncio.new_event_loop()
                        try:
                            loop.run_until_complete(
                                add_episode(
                                    content=content,
                                    agent_id=agent,
                                    source=f"lcm:{conv_id}",
                                )
                            )
                            graphiti_fed += 1
                        finally:
                            loop.close()
                    except Exception as e:
                        print(f"[{datetime.utcnow()}] Graphiti feed error: {e}", file=sys.stderr)

            except Exception as e:
                print(f"[{datetime.utcnow()}] Error adding memory: {e}", file=sys.stderr)

            last_id = max(last_id, s.get("rowid", 0))

        state["last_summary_id"] = last_id
        state["last_sync"] = datetime.utcnow().isoformat()
        state["total_extracted"] = state.get("total_extracted", 0) + imported
        state["total_graphiti_fed"] = state.get("total_graphiti_fed", 0) + graphiti_fed
        save_json(state, STATE_FILE)
        return imported


def main():
    """Run the watcher loop."""
    print(f"[{datetime.utcnow()}] MemClawz v6 watcher starting (interval={SYNC_INTERVAL}s)")
    watcher = LCMWatcher()
    while True:
        try:
            n = watcher.sync()
            if n > 0:
                print(f"[{datetime.utcnow()}] Extracted {n} new memories")
        except Exception as e:
            print(f"[{datetime.utcnow()}] Sync error: {e}", file=sys.stderr)
        time.sleep(SYNC_INTERVAL)


if __name__ == "__main__":
    main()
