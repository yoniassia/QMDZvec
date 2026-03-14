"""MemClawz v7 — Phase 5: Hybrid Search

Implements BM25 keyword scoring alongside vector search for improved relevance.
Pure Python implementation with no external dependencies.
"""
import math
import re
import logging
from typing import List, Dict, Any, Set
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

# Default BM25 parameters (standard values)
DEFAULT_K1 = 1.5  # term frequency saturation point
DEFAULT_B = 0.75  # length normalization parameter

# Default hybrid search weights
DEFAULT_HYBRID_WEIGHTS = {
    "w_semantic": 0.40,   # vector similarity weight
    "w_keyword": 0.15,    # BM25 keyword weight  
    "w_recency": 0.25,    # recency decay weight
    "w_importance": 0.20, # importance/type weight
}

# Status weight multipliers for lifecycle integration
STATUS_WEIGHTS = {
    "confirmed": 1.2,     # high confidence boost
    "active": 1.0,        # baseline
    "outdated": 0.5,      # reduced relevance
    "archived": 0.3,      # minimal relevance
    "contradicted": 0.1,  # very low relevance
    "merged": 0.2,        # low relevance (superseded)
    "superseded": 0.2,    # low relevance (superseded)
    "deleted": 0.0,       # no relevance
}


def tokenize(text: str) -> List[str]:
    """Simple tokenization for BM25.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of lowercase tokens
    """
    if not text:
        return []
    
    # Convert to lowercase and split on non-alphanumeric characters
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    
    # Filter out very short tokens
    return [token for token in tokens if len(token) >= 2]


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Compute term frequency for a document.
    
    Args:
        tokens: List of tokens from the document
        
    Returns:
        Dict mapping term -> frequency
    """
    if not tokens:
        return {}
    
    counts = Counter(tokens)
    doc_length = len(tokens)
    
    return {term: count / doc_length for term, count in counts.items()}


def compute_idf(documents: List[List[str]]) -> Dict[str, float]:
    """Compute inverse document frequency for a corpus.
    
    Args:
        documents: List of tokenized documents
        
    Returns:
        Dict mapping term -> IDF score
    """
    if not documents:
        return {}
    
    N = len(documents)
    term_doc_count = defaultdict(int)
    
    # Count documents containing each term
    for doc_tokens in documents:
        unique_terms = set(doc_tokens)
        for term in unique_terms:
            term_doc_count[term] += 1
    
    # Compute IDF: log(N / df_t) where df_t is document frequency of term t
    idf = {}
    for term, df in term_doc_count.items():
        idf[term] = math.log(N / df) if df > 0 else 0.0
    
    return idf


def bm25_score(
    query: str, 
    document: str, 
    corpus_stats: Dict[str, Any] = None,
    k1: float = DEFAULT_K1, 
    b: float = DEFAULT_B
) -> float:
    """Compute BM25 score for a query-document pair.
    
    Args:
        query: Search query string
        document: Document text to score
        corpus_stats: Pre-computed corpus statistics (idf, avgdl) or None
        k1: BM25 k1 parameter (term frequency saturation)
        b: BM25 b parameter (length normalization)
        
    Returns:
        BM25 score (higher = more relevant)
    """
    if not query or not document:
        return 0.0
    
    query_tokens = tokenize(query)
    doc_tokens = tokenize(document)
    
    if not query_tokens or not doc_tokens:
        return 0.0
    
    doc_length = len(doc_tokens)
    doc_tf = compute_tf(doc_tokens)
    
    # If no corpus stats provided, compute minimal IDF
    if corpus_stats is None:
        # Simple fallback: assume all query terms are relatively rare
        idf_scores = {term: 2.0 for term in set(query_tokens)}
        avgdl = doc_length  # assume this doc is average length
    else:
        idf_scores = corpus_stats.get("idf", {})
        avgdl = corpus_stats.get("avgdl", doc_length)
    
    score = 0.0
    
    # Sum BM25 score for each query term
    for term in query_tokens:
        if term not in doc_tf:
            continue
            
        tf = doc_tf[term] * len(doc_tokens)  # convert back to raw count
        idf = idf_scores.get(term, 1.0)
        
        # BM25 formula
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_length / avgdl))
        
        score += idf * (numerator / denominator)
    
    return score


def prepare_corpus_stats(documents: List[str]) -> Dict[str, Any]:
    """Pre-compute corpus statistics for efficient BM25 scoring.
    
    Args:
        documents: List of document texts
        
    Returns:
        Dict with 'idf' and 'avgdl' keys
    """
    if not documents:
        return {"idf": {}, "avgdl": 0}
    
    tokenized_docs = [tokenize(doc) for doc in documents]
    
    # Compute IDF for all terms in corpus
    idf = compute_idf(tokenized_docs)
    
    # Compute average document length
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
    
    return {
        "idf": idf,
        "avgdl": avgdl,
        "doc_count": len(documents)
    }


def hybrid_search(
    query: str,
    vector_results: List[Dict[str, Any]], 
    top_k: int = 20,
    weights: Dict[str, float] = None,
    corpus_stats: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """Combine vector search results with BM25 keyword scoring.
    
    Args:
        query: Original search query
        vector_results: Results from vector/semantic search
        top_k: Number of results to return
        weights: Hybrid scoring weights (w_semantic, w_keyword, w_recency, w_importance)
        corpus_stats: Pre-computed corpus statistics for BM25
        
    Returns:
        Re-ranked results with hybrid scores
    """
    if not vector_results:
        return []
    
    # Use default weights if not provided
    if weights is None:
        weights = DEFAULT_HYBRID_WEIGHTS.copy()
    
    scored_results = []
    
    # Extract documents for corpus stats if not provided
    if corpus_stats is None:
        documents = []
        for result in vector_results:
            doc_text = result.get("memory", result.get("content", ""))
            if doc_text:
                documents.append(doc_text)
        corpus_stats = prepare_corpus_stats(documents)
    
    for result in vector_results:
        # Get document content
        doc_content = result.get("memory", result.get("content", ""))
        
        # Get vector similarity score
        semantic_score = result.get("score", result.get("composite_score", 0.5))
        
        # Compute BM25 keyword score
        keyword_score = bm25_score(query, doc_content, corpus_stats)
        
        # Normalize BM25 score (rough normalization)
        # BM25 scores can vary widely, so we apply a sigmoid-like normalization
        normalized_keyword = 1.0 / (1.0 + math.exp(-keyword_score / 3.0))
        
        # Get metadata for additional scoring factors
        metadata = result.get("metadata", {})
        payload = result.get("payload", metadata)  # handle both Qdrant and Mem0 formats
        
        # Recency and importance from existing composite scoring
        from .scoring import composite_score
        
        # Extract fields for composite scoring
        created_at = payload.get("created_at", metadata.get("created_at"))
        importance = payload.get("importance", metadata.get("importance", 0.8))
        access_count = payload.get("access_count", metadata.get("access_count", 0))
        memory_type = payload.get("memory_type", metadata.get("type", "fact"))
        
        # Get recency component from composite scoring
        composite = composite_score(
            semantic_similarity=semantic_score,
            created_at=created_at,
            importance=importance,
            access_count=access_count,
            memory_type=memory_type
        )
        
        # Extract individual components (approximate)
        # This is a simplified extraction - ideally we'd refactor composite_score
        # to return individual components
        from .scoring import _compute_recency, TYPE_BOOST
        recency = _compute_recency(created_at)
        type_weight = TYPE_BOOST.get(memory_type, 0.8)
        weighted_importance = importance * type_weight
        
        # Apply lifecycle status weight
        status = payload.get("status", "active")
        status_multiplier = STATUS_WEIGHTS.get(status, 1.0)
        
        # Compute hybrid score
        hybrid_score = (
            weights["w_semantic"] * semantic_score +
            weights["w_keyword"] * normalized_keyword +
            weights["w_recency"] * recency +
            weights["w_importance"] * weighted_importance
        ) * status_multiplier
        
        # Clamp to [0, 1]
        hybrid_score = max(0.0, min(1.0, hybrid_score))
        
        # Add scores to result
        enhanced_result = result.copy()
        enhanced_result.update({
            "hybrid_score": hybrid_score,
            "keyword_score": normalized_keyword,
            "raw_bm25": keyword_score,
            "recency_score": recency,
            "importance_score": weighted_importance,
            "status_multiplier": status_multiplier,
            "status": status,
        })
        
        scored_results.append(enhanced_result)
    
    # Sort by hybrid score (descending)
    scored_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    # Return top K results
    return scored_results[:top_k]


def explain_hybrid_score(result: Dict[str, Any], query: str = "") -> Dict[str, Any]:
    """Explain how a hybrid score was computed for debugging.
    
    Args:
        result: Result dict with hybrid scoring components
        query: Original query (for context)
        
    Returns:
        Dict with scoring breakdown
    """
    return {
        "query": query,
        "content_preview": result.get("memory", result.get("content", ""))[:100] + "...",
        "final_score": result.get("hybrid_score", 0),
        "components": {
            "semantic": result.get("score", 0),
            "keyword_normalized": result.get("keyword_score", 0),
            "raw_bm25": result.get("raw_bm25", 0),
            "recency": result.get("recency_score", 0),
            "importance": result.get("importance_score", 0),
        },
        "multipliers": {
            "status": result.get("status_multiplier", 1.0),
            "status_name": result.get("status", "active"),
        },
        "metadata": {
            "memory_type": result.get("metadata", {}).get("memory_type", "unknown"),
            "agent_id": result.get("metadata", {}).get("agent_id", "unknown"),
            "created_at": result.get("metadata", {}).get("created_at", "unknown"),
        }
    }


def batch_hybrid_search(
    queries: List[str],
    vector_results_list: List[List[Dict[str, Any]]], 
    top_k: int = 20,
    weights: Dict[str, float] = None
) -> List[List[Dict[str, Any]]]:
    """Perform hybrid search on multiple query-result pairs efficiently.
    
    Args:
        queries: List of search queries
        vector_results_list: List of vector search results (one per query)
        top_k: Number of results to return per query
        weights: Hybrid scoring weights
        
    Returns:
        List of re-ranked results (one per query)
    """
    if len(queries) != len(vector_results_list):
        raise ValueError("queries and vector_results_list must have same length")
    
    results = []
    
    # Pre-compute corpus stats from all documents for consistency
    all_documents = []
    for vector_results in vector_results_list:
        for result in vector_results:
            doc_text = result.get("memory", result.get("content", ""))
            if doc_text:
                all_documents.append(doc_text)
    
    corpus_stats = prepare_corpus_stats(all_documents)
    
    # Process each query
    for query, vector_results in zip(queries, vector_results_list):
        hybrid_results = hybrid_search(
            query=query,
            vector_results=vector_results,
            top_k=top_k,
            weights=weights,
            corpus_stats=corpus_stats
        )
        results.append(hybrid_results)
    
    return results