"""MemClawz v6 — Compaction Cron.

Runs every 30 minutes:
1. Check memory stats and health
2. Run daily digest at midnight UTC
3. Run weekly merge on Sundays at midnight UTC
4. Trigger reflection when appropriate
"""
import logging
import time
import sys
from datetime import datetime, timezone

from .config import STATE_DIR, COMPACTION_INTERVAL
from .utils import load_json, save_json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

CRON_STATE_FILE = STATE_DIR / "compaction_cron_state.json"


def run_cycle():
    """Run one compaction cycle."""
    from mem0 import Memory
    from .config import MEM0_CONFIG
    from .compactor import DailyDigest, WeeklyMerge, get_compaction_status
    from .reflection import ReflectionEngine

    mem = Memory.from_config(MEM0_CONFIG)
    now = datetime.now(timezone.utc)
    state = load_json(CRON_STATE_FILE)
    today = now.strftime("%Y-%m-%d")
    actions = []

    # --- Daily Digest (run once per day, after midnight UTC) ---
    last_daily = state.get("last_daily_date", "")
    if last_daily != today and now.hour >= 0:
        logger.info(f"Running daily digest for {today}")
        try:
            digest = DailyDigest(mem)
            result = digest.generate(today)
            actions.append(f"daily_digest: {result.get('status')}, {result.get('memories_added', 0)} memories added")
            state["last_daily_date"] = today
        except Exception as e:
            logger.error(f"Daily digest error: {e}")
            actions.append(f"daily_digest: error - {e}")

    # --- Weekly Merge (run once per week, Sunday) ---
    last_weekly = state.get("last_weekly_date", "")
    if now.weekday() == 6 and last_weekly != today:
        logger.info("Running weekly merge")
        try:
            merger = WeeklyMerge(mem)
            result = merger.merge()
            actions.append(f"weekly_merge: {result.get('status')}, {result.get('dedup_removed', 0)} duplicates removed")
            state["last_weekly_date"] = today
        except Exception as e:
            logger.error(f"Weekly merge error: {e}")
            actions.append(f"weekly_merge: error - {e}")

    # --- Reflection (run once per day, in off-hours) ---
    last_reflect = state.get("last_reflect_date", "")
    if last_reflect != today and now.hour >= 2 and now.hour <= 5:
        logger.info("Running sleep-time reflection")
        try:
            engine = ReflectionEngine(mem)
            result = engine.reflect(hours=24)
            actions.append(f"reflection: {result.get('status')}, {result.get('insights_stored', 0)} insights")
            state["last_reflect_date"] = today
        except Exception as e:
            logger.error(f"Reflection error: {e}")
            actions.append(f"reflection: error - {e}")

    # --- Health stats ---
    try:
        status = get_compaction_status()
        state["last_health"] = {
            "session_files": status.get("session_files", 0),
            "daily_digests": status.get("daily_digests", 0),
            "archived_files": status.get("archived_files", 0),
        }
    except Exception:
        pass

    state["last_run"] = now.isoformat()
    state["last_actions"] = actions
    state["runs_total"] = state.get("runs_total", 0) + 1
    save_json(state, CRON_STATE_FILE)

    if actions:
        logger.info(f"Cron cycle completed: {actions}")
    else:
        logger.info("Cron cycle completed: no actions needed")


def main():
    """Run the compaction cron loop."""
    logger.info(f"MemClawz compaction cron starting (interval={COMPACTION_INTERVAL}s)")
    while True:
        try:
            run_cycle()
        except Exception as e:
            logger.error(f"Cron cycle error: {e}")
        time.sleep(COMPACTION_INTERVAL)


if __name__ == "__main__":
    main()
