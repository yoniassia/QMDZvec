"""MemClawz v7 — Phase 4: Memory Status Lifecycle

Implements memory status lifecycle management with 8 states and transition validation.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, Range, MatchValue
from .config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME

logger = logging.getLogger(__name__)

# 8 lifecycle states
LIFECYCLE_STATES = {
    "active",      # default - actively used memory
    "confirmed",   # high confidence - accessed multiple times, still valid
    "outdated",    # potentially stale - needs validation
    "archived",    # preserved but not actively used
    "contradicted", # contradicted by newer information
    "merged",      # combined into another memory
    "superseded",  # replaced by newer version
    "deleted",     # marked for deletion
}

# Valid state transitions
VALID_TRANSITIONS = {
    "active": {"confirmed", "outdated", "archived", "contradicted", "merged", "superseded", "deleted"},
    "confirmed": {"outdated", "archived", "superseded", "deleted"},
    "outdated": {"archived", "deleted", "active"},  # can be re-validated
    "archived": {"active", "deleted"},  # can be restored
    "contradicted": {"deleted", "active"},  # can be re-validated
    "merged": {"deleted"},
    "superseded": {"deleted"},
    "deleted": set(),  # final state
}

# Default status for new memories
DEFAULT_STATUS = "active"


class MemoryLifecycle:
    """Memory lifecycle management with status tracking and transitions."""
    
    def __init__(self):
        self.qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    def transition(self, memory_id: str, from_status: str, to_status: str) -> bool:
        """Validate and execute a status transition.
        
        Args:
            memory_id: UUID of the memory to transition
            from_status: Expected current status (for validation)
            to_status: Target status
            
        Returns:
            True if transition succeeded, False if invalid or failed
        """
        # Validate transition is allowed
        if from_status not in VALID_TRANSITIONS:
            logger.warning(f"Invalid from_status: {from_status}")
            return False
            
        if to_status not in VALID_TRANSITIONS[from_status]:
            logger.warning(f"Invalid transition: {from_status} -> {to_status}")
            return False
        
        try:
            # Get current memory to verify status
            result = self.qc.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[memory_id]
            )
            
            if not result:
                logger.warning(f"Memory {memory_id} not found")
                return False
                
            current_payload = result[0].payload
            current_status = current_payload.get("status", DEFAULT_STATUS)
            
            # Verify current status matches expected from_status
            if current_status != from_status:
                logger.warning(f"Status mismatch: expected {from_status}, got {current_status}")
                return False
            
            # Update status
            updated_payload = current_payload.copy()
            updated_payload["status"] = to_status
            updated_payload["status_updated_at"] = datetime.now(timezone.utc).isoformat()
            
            # Add transition metadata
            if to_status in {"superseded", "contradicted", "merged"}:
                updated_payload["lifecycle_metadata"] = updated_payload.get("lifecycle_metadata", {})
                updated_payload["lifecycle_metadata"]["transition_timestamp"] = datetime.now(timezone.utc).isoformat()
            
            # Update payload in Qdrant (use set_payload to avoid needing the vector)
            self.qc.set_payload(
                collection_name=COLLECTION_NAME,
                payload={
                    "status": to_status,
                    "status_updated_at": updated_payload["status_updated_at"],
                    **({"lifecycle_metadata": updated_payload["lifecycle_metadata"]} if "lifecycle_metadata" in updated_payload else {})
                },
                points=[memory_id]
            )
            
            logger.info(f"Memory {memory_id} transitioned: {from_status} -> {to_status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to transition memory {memory_id}: {e}")
            return False
    
    def get_status(self, memory_id: str) -> str:
        """Get the current status of a memory.
        
        Args:
            memory_id: UUID of the memory
            
        Returns:
            Current status string, or DEFAULT_STATUS if not found
        """
        try:
            result = self.qc.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[memory_id]
            )
            
            if not result:
                logger.warning(f"Memory {memory_id} not found")
                return DEFAULT_STATUS
                
            return result[0].payload.get("status", DEFAULT_STATUS)
            
        except Exception as e:
            logger.error(f"Failed to get status for memory {memory_id}: {e}")
            return DEFAULT_STATUS
    
    def bulk_check_outdated(self, threshold_days: int = 30) -> List[Dict[str, Any]]:
        """Find memories that should be marked as outdated.
        
        Args:
            threshold_days: Number of days after which a memory is considered outdated
            
        Returns:
            List of memory dicts with id, content, age_days, current_status
        """
        threshold_date = datetime.now(timezone.utc) - timedelta(days=threshold_days)
        threshold_iso = threshold_date.isoformat()
        
        try:
            # Search for active memories older than threshold
            filter_conditions = Filter(
                must=[
                    FieldCondition(
                        key="status",
                        match=MatchValue(value="active")
                    ),
                    FieldCondition(
                        key="created_at",
                        range=Range(lt=threshold_iso)
                    )
                ]
            )
            
            results, _ = self.qc.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=filter_conditions,
                limit=1000,
                with_payload=True,
            )
            
            outdated_candidates = []
            
            for hit in results:
                payload = hit.payload
                created_str = payload.get("created_at", "")
                
                if created_str:
                    try:
                        created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                        age_days = (datetime.now(timezone.utc) - created_dt.replace(
                            tzinfo=timezone.utc if created_dt.tzinfo is None else created_dt.tzinfo
                        )).days
                        
                        outdated_candidates.append({
                            "id": hit.id,
                            "content": payload.get("memory", "")[:100] + "...",  # truncated preview
                            "age_days": age_days,
                            "current_status": payload.get("status", DEFAULT_STATUS),
                            "memory_type": payload.get("memory_type", "unknown"),
                            "agent_id": payload.get("agent_id", "unknown")
                        })
                    except (ValueError, TypeError):
                        continue
            
            logger.info(f"Found {len(outdated_candidates)} memories older than {threshold_days} days")
            return outdated_candidates
            
        except Exception as e:
            logger.error(f"Failed to check for outdated memories: {e}")
            return []
    
    def confirm(self, memory_id: str) -> bool:
        """Mark a memory as confirmed (high confidence, accessed multiple times).
        
        Args:
            memory_id: UUID of the memory to confirm
            
        Returns:
            True if successful, False if failed
        """
        current_status = self.get_status(memory_id)
        if current_status == "confirmed":
            return True  # already confirmed
            
        return self.transition(memory_id, current_status, "confirmed")
    
    def supersede(self, old_id: str, new_id: str) -> bool:
        """Mark an old memory as superseded by a new one.
        
        Args:
            old_id: UUID of the memory being superseded
            new_id: UUID of the superseding memory
            
        Returns:
            True if successful, False if failed
        """
        try:
            old_status = self.get_status(old_id)
            
            # Transition old memory to superseded
            if not self.transition(old_id, old_status, "superseded"):
                return False
            
            # Add supersession link metadata to old memory
            self.qc.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"lifecycle_metadata": {"superseded_by": new_id}},
                points=[old_id]
            )
            
            logger.info(f"Memory {old_id} superseded by {new_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to supersede memory {old_id}: {e}")
            return False
    
    def contradict(self, memory_id: str, contradicting_id: str) -> bool:
        """Mark a memory as contradicted by another.
        
        Args:
            memory_id: UUID of the memory being contradicted
            contradicting_id: UUID of the contradicting memory
            
        Returns:
            True if successful, False if failed
        """
        try:
            current_status = self.get_status(memory_id)
            
            # Transition to contradicted
            if not self.transition(memory_id, current_status, "contradicted"):
                return False
            
            # Add contradiction link metadata
            self.qc.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"lifecycle_metadata": {"contradicted_by": contradicting_id}},
                points=[memory_id]
            )
            
            logger.info(f"Memory {memory_id} contradicted by {contradicting_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to contradict memory {memory_id}: {e}")
            return False
    
    def get_lifecycle_stats(self) -> Dict[str, int]:
        """Get counts of memories by lifecycle status.
        
        Returns:
            Dict mapping status -> count
        """
        stats = {}
        
        try:
            for status in LIFECYCLE_STATES:
                filter_condition = Filter(
                    must=[FieldCondition(key="status", match=MatchValue(value=status))]
                )
                
                count_result = self.qc.count(
                    collection_name=COLLECTION_NAME,
                    count_filter=filter_condition,
                    exact=True,
                )
                stats[status] = count_result.count
            
            # Total collection count for "no status" estimation
            total = self.qc.count(collection_name=COLLECTION_NAME, exact=True).count
            labeled = sum(stats.values())
            stats["unlabeled"] = max(0, total - labeled)
            
        except Exception as e:
            logger.error(f"Failed to get lifecycle stats: {e}")
        
        return stats