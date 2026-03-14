#!/usr/bin/env bash
#
# MemClawz daily fleet sync
# -------------------------
# Pulls OpenClaw conversation/memory data from remote servers and POSTs
# extracted content to localhost:3500/api/v1/add with proper agent_id tags.
#
# Usage:
#   ./scripts/fleet-sync.sh [API_BASE_URL]
#   Default API_BASE_URL is http://localhost:3500
#
# Hosts synced:
#   - White Rabbit:  ssh root@135.181.43.68
#   - Clawdet:       ssh root@188.34.197.212
#   - claw-fleet-ams: ssh -i ~/.ssh/oci_claw_fleet ubuntu@141.144.203.233
#   - MoneyClaw:     skipped (SSH access denied)
#
# Requires: bash, ssh, find, python3, curl
# Run daily via cron or systemd timer (not installed by this script).
#
set -euo pipefail

API_BASE="${1:-http://localhost:3500}"
LOG_PREFIX="[fleet-sync]"
TMP_DIR=""
ADD_ENDPOINT="${API_BASE}/api/v1/add"

# Host list: "display_name|ssh_invocation"
HOSTS=(
  "White Rabbit|ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new root@135.181.43.68"
  "Clawdet|ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new root@188.34.197.212"
  "claw-fleet-ams|ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -i ~/.ssh/oci_claw_fleet ubuntu@141.144.203.233"
)

log() { echo "$LOG_PREFIX $*"; }
err() { echo "$LOG_PREFIX ERROR: $*" >&2; }

cleanup() {
  [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]] && rm -rf "$TMP_DIR"
}
trap cleanup EXIT
TMP_DIR="$(mktemp -d)"

# Build JSON payload and POST to MemClawz add endpoint.
# Usage: post_add <content_file> "agent_id" "memory_type" "source_host" "remote_path"
# Content read from file to avoid ARG_MAX for large files.
post_add() {
  local content_file="$1" agent_id="$2" memory_type="$3" source_host="$4" remote_path="$5"
  python3 - "$content_file" "$agent_id" "$memory_type" "$source_host" "$remote_path" << 'PY' | curl -sf -X POST -H "Content-Type: application/json" -d @- "$ADD_ENDPOINT" >/dev/null
import sys, json
from pathlib import Path
content_file = sys.argv[1]
agent_id = sys.argv[2]
memory_type = sys.argv[3]
source_host = sys.argv[4]
remote_path = sys.argv[5]
content = Path(content_file).read_text()
payload = {
    "content": content,
    "user_id": "yoni",
    "agent_id": agent_id,
    "memory_type": memory_type,
    "metadata": {
        "source": "fleet_sync",
        "source_host": source_host,
        "remote_path": remote_path,
    },
}
print(json.dumps(payload))
PY
}

# Derive agent_id from remote path.
# Sessions: .../agents/<agent>/sessions/...
# Memory:   .../agents-live/<agent>/memory/... or .../agents-live/<agent>/MEMORY.md
agent_from_path() {
  local path="$1" kind="$2"
  if [[ "$kind" == "session" ]]; then
    if [[ "$path" =~ agents/([^/]+)/sessions/ ]]; then
      echo "${BASH_REMATCH[1]}"
      return
    fi
  else
    if [[ "$path" =~ agents-live/([^/]+)/ ]]; then
      echo "${BASH_REMATCH[1]}"
      return
    fi
  fi
  echo "unknown"
}

sync_host() {
  local name="$1" ssh_cmd="$2"
  log "Syncing host: $name"
  local session_files memory_files
  session_files="$TMP_DIR/sessions_$$.txt"
  memory_files="$TMP_DIR/memory_$$.txt"

  if ! $ssh_cmd "find ~/.openclaw/agents -path '*/sessions/*.jsonl' 2>/dev/null" > "$session_files" 2>/dev/null; then
    err "Failed to discover session files on $name; skipping host."
    return 0
  fi
  if ! $ssh_cmd "{ find ~/.openclaw/workspace/agents-live -path '*/memory/*.md' 2>/dev/null; find ~/.openclaw/workspace/agents-live -maxdepth 3 -name MEMORY.md 2>/dev/null; }" > "$memory_files" 2>/dev/null; then
    err "Failed to discover memory files on $name; continuing with sessions only."
    : > "$memory_files"
  fi

  local count=0
  while IFS= read -r rem_path; do
    [[ -z "$rem_path" ]] && continue
    local tmpf="$TMP_DIR/file_$$"
    if ! $ssh_cmd "cat $(printf '%q' "$rem_path")" 2>/dev/null > "$tmpf"; then
      err "Failed to read $name:$rem_path; skipping file."
      continue
    fi
    local agent_id
    agent_id="$(agent_from_path "$rem_path" "session")"
    if [[ ! -s "$tmpf" ]]; then
      log "Empty file $name:$rem_path; skipping."
      rm -f "$tmpf"
      continue
    fi
    if ! post_add "$tmpf" "$agent_id" "transcript" "$name" "$rem_path"; then
      rm -f "$tmpf"
      err "POST failed for $name:$rem_path; skipping."
      continue
    fi
    rm -f "$tmpf"
    (( count++ )) || true
  done < "$session_files"

  while IFS= read -r rem_path; do
    [[ -z "$rem_path" ]] && continue
    local tmpf="$TMP_DIR/file_$$"
    if ! $ssh_cmd "cat $(printf '%q' "$rem_path")" 2>/dev/null > "$tmpf"; then
      err "Failed to read $name:$rem_path; skipping file."
      continue
    fi
    local agent_id
    agent_id="$(agent_from_path "$rem_path" "memory")"
    if [[ ! -s "$tmpf" ]]; then
      log "Empty file $name:$rem_path; skipping."
      rm -f "$tmpf"
      continue
    fi
    if ! post_add "$tmpf" "$agent_id" "memory" "$name" "$rem_path"; then
      rm -f "$tmpf"
      err "POST failed for $name:$rem_path; skipping."
      continue
    fi
    rm -f "$tmpf"
    (( count++ )) || true
  done < "$memory_files"

  log "Host $name: posted $count items."
}

main() {
  log "Starting fleet sync → $ADD_ENDPOINT"
  for entry in "${HOSTS[@]}"; do
    name="${entry%%|*}"
    ssh_cmd="${entry#*|}"
    if ! sync_host "$name" "$ssh_cmd"; then
      err "Host $name failed; continuing to next host."
    fi
  done
  log "Fleet sync finished."
}

main "$@"
