"""MemClawz v7 — Extension functions for main API integration

This module exports helper functions that can be imported by api.py
to integrate v7 features (lifecycle, hybrid search, expanded types)
without directly modifying the main API file.
"""
import logging
from typing import List, Dict, Any, Optional

from .lifecycle import MemoryLifecycle, DEFAULT_STATUS
from .hybrid_search import hybrid_search, prepare_corpus_stats, DEFAULT_HYBRID_WEIGHTS, STATUS_WEIGHTS

logger = logging.getLogger(__name__)

# Re-export status weights for easy import
__all__ = [
    "STATUS_WEIGHTS",
    "apply_lifecycle_filter", 
    "apply_hybrid_scoring",
    "get_lifecycle_manager",
    "enhance_memory_with_lifecycle",
    "get_v7_stats"
]


def apply_lifecycle_filter(
    results: List[Dict[str, Any]], 
    include_deleted: bool = False,
    include_contradicted: bool = True,
    include_superseded: bool = False
) -> List[Dict[str, Any]]:
    """Filter search results by lifecycle status.
    
    Args:
        results: List of search results from Qdrant/Mem0
        include_deleted: Whether to include deleted memories  
        include_contradicted: Whether to include contradicted memories
        include_superseded: Whether to include superseded memories
        
    Returns:
        Filtered results list
    """
    if not results:
        return []
    
    filtered = []
    
    for result in results:
        # Extract status from payload or metadata
        metadata = result.get("metadata", {})
        payload = result.get("payload", metadata)
        status = payload.get("status", DEFAULT_STATUS)
        
        # Apply filtering rules
        if status == "deleted" and not include_deleted:
            continue
        elif status == "contradicted" and not include_contradicted:
            continue
        elif status == "superseded" and not include_superseded:
            continue
        
        filtered.append(result)
    
    logger.debug(f"Lifecycle filter: {len(results)} -> {len(filtered)} results")
    return filtered


def apply_hybrid_scoring(
    query: str, 
    results: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """Apply hybrid scoring (BM25 + lifecycle + composite) to search results.
    
    Args:
        query: Original search query
        results: Vector search results
        weights: Hybrid scoring weights (optional)
        top_k: Number of results to return
        
    Returns:
        Re-scored and re-ranked results
    """
    if not results or not query.strip():
        return results[:top_k]
    
    try:
        # Use hybrid search to re-score
        hybrid_results = hybrid_search(
            query=query,
            vector_results=results,
            top_k=top_k,
            weights=weights or DEFAULT_HYBRID_WEIGHTS
        )
        
        logger.debug(f"Hybrid scoring: re-ranked {len(results)} -> {len(hybrid_results)} results")
        return hybrid_results
        
    except Exception as e:
        logger.error(f"Hybrid scoring failed: {e}")
        # Fallback to original results
        return results[:top_k]


def get_lifecycle_manager() -> MemoryLifecycle:
    """Get a MemoryLifecycle instance for status management.
    
    Returns:
        Configured MemoryLifecycle instance
    """
    return MemoryLifecycle()


def enhance_memory_with_lifecycle(memory_data: Dict[str, Any]) -> Dict[str, Any]:
    """Add lifecycle fields to a new memory before storage.
    
    Args:
        memory_data: Memory data dict (for _direct_qdrant_upsert payload)
        
    Returns:
        Enhanced memory data with lifecycle fields
    """
    enhanced = memory_data.copy()
    
    # Add default lifecycle status if not present
    if "status" not in enhanced:
        enhanced["status"] = DEFAULT_STATUS
        
    # Add status timestamp
    from datetime import datetime, timezone
    enhanced["status_updated_at"] = datetime.now(timezone.utc).isoformat()
    
    # Initialize lifecycle metadata
    if "lifecycle_metadata" not in enhanced:
        enhanced["lifecycle_metadata"] = {}
    
    return enhanced


def get_v7_stats() -> Dict[str, Any]:
    """Get v7 feature statistics and health info.
    
    Returns:
        Dict with lifecycle stats, hybrid search info, etc.
    """
    try:
        lifecycle = get_lifecycle_manager()
        lifecycle_stats = lifecycle.get_lifecycle_stats()
        
        return {
            "version": "7.0",
            "features": {
                "lifecycle": True,
                "hybrid_search": True,
                "expanded_types": True,
            },
            "lifecycle_stats": lifecycle_stats,
            "memory_types": {
                "v6_types": [
                    "decision", "preference", "relationship", 
                    "insight", "procedure", "fact", "event"
                ],
                "v7_types": [
                    "intention", "plan", "commitment", 
                    "action", "outcome", "cancellation"
                ],
                "persistent_types": [
                    "decision", "preference", "relationship", "commitment"
                ]
            },
            "status_weights": STATUS_WEIGHTS,
            "hybrid_weights": DEFAULT_HYBRID_WEIGHTS,
        }
        
    except Exception as e:
        logger.error(f"Failed to get v7 stats: {e}")
        return {
            "version": "7.0",
            "features": {
                "lifecycle": False,
                "hybrid_search": True,  # pure Python, should always work
                "expanded_types": True,  # just dict updates
            },
            "error": str(e)
        }


def process_search_v7(
    query: str,
    raw_results: List[Dict[str, Any]],
    apply_lifecycle: bool = True,
    apply_hybrid: bool = True,
    include_deleted: bool = False,
    top_k: int = 20,
    weights: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """Process search results with all v7 enhancements.
    
    This is a convenience function that applies:
    1. Lifecycle filtering
    2. Hybrid scoring  
    3. Re-ranking
    
    Args:
        query: Search query string
        raw_results: Raw vector search results
        apply_lifecycle: Whether to apply lifecycle filtering
        apply_hybrid: Whether to apply hybrid scoring
        include_deleted: Whether to include deleted memories in results
        top_k: Number of results to return
        weights: Hybrid scoring weights
        
    Returns:
        Processed and ranked results
    """
    results = raw_results
    
    # Step 1: Lifecycle filtering
    if apply_lifecycle:
        results = apply_lifecycle_filter(
            results, 
            include_deleted=include_deleted,
            include_contradicted=True,  # usually want to see contradicted
            include_superseded=False    # usually don't want superseded
        )
    
    # Step 2: Hybrid scoring and re-ranking
    if apply_hybrid and query.strip():
        results = apply_hybrid_scoring(
            query=query,
            results=results,
            weights=weights,
            top_k=top_k
        )
    else:
        # Just truncate to top_k
        results = results[:top_k]
    
    return results


def transition_memory_status(
    memory_id: str, 
    from_status: str, 
    to_status: str
) -> Dict[str, Any]:
    """Transition a memory's lifecycle status with error handling.
    
    Args:
        memory_id: Memory UUID
        from_status: Expected current status
        to_status: Target status
        
    Returns:
        Dict with success/error info
    """
    try:
        lifecycle = get_lifecycle_manager()
        success = lifecycle.transition(memory_id, from_status, to_status)
        
        if success:
            return {
                "status": "success",
                "message": f"Transitioned {memory_id}: {from_status} -> {to_status}",
                "memory_id": memory_id,
                "from_status": from_status,
                "to_status": to_status
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to transition {memory_id}: {from_status} -> {to_status}",
                "memory_id": memory_id,
                "from_status": from_status,
                "to_status": to_status
            }
            
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Exception during transition: {e}",
            "memory_id": memory_id,
            "error": str(e)
        }


def bulk_update_outdated(threshold_days: int = 30) -> Dict[str, Any]:
    """Find and mark outdated memories in bulk.
    
    Args:
        threshold_days: Age threshold for marking as outdated
        
    Returns:
        Dict with operation results
    """
    try:
        lifecycle = get_lifecycle_manager()
        candidates = lifecycle.bulk_check_outdated(threshold_days)
        
        updated_count = 0
        errors = []
        
        for candidate in candidates:
            memory_id = candidate["id"]
            current_status = candidate["current_status"]
            
            # Only transition active memories to outdated
            if current_status == "active":
                success = lifecycle.transition(memory_id, "active", "outdated")
                if success:
                    updated_count += 1
                else:
                    errors.append(f"Failed to update {memory_id}")
        
        return {
            "status": "success",
            "candidates_found": len(candidates),
            "updated_count": updated_count,
            "errors": errors,
            "threshold_days": threshold_days
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Bulk update failed: {e}",
            "error": str(e)
        }