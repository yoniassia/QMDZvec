#!/usr/bin/env python3
"""
Re-attribute memories with missing agent_id field in MemClawz Qdrant
"""

from qdrant_client import QdrantClient
import json
import time
import re
from typing import Dict, List, Optional

# Agent routing rules (classify by content keywords)
AGENT_KEYWORDS = {
    'infraclaw': [
        'server', 'deploy', 'DNS', 'Caddy', 'Docker', 'SSH', 'firewall', 
        'Hetzner', 'systemd', 'nginx', 'SSL', 'port', 'infrastructure',
        'deployment', 'hosting', 'domain', 'certificate'
    ],
    'tradeclaw': [
        'trade', 'portfolio', 'market', 'stock', 'ETF', 'DeFi', 'ClawMM', 
        'Hyperliquid', 'brief', 'holdings', 'BTC', 'crypto price', 'trading',
        'exchange', 'position', 'profit', 'loss', 'order'
    ],
    'appsclaw': [
        'app store', 'swagger', 'plugin', 'eToro app', 'liquidation map', 
        'dashboard app', 'application', 'frontend', 'UI', 'interface'
    ],
    'tradingdataclaw': [
        'data module', 'AlphaEar', 'DataScout', 'quant', 'signal', 
        'backtest', 'VectorBT', 'MCP data', 'analytics', 'metrics',
        'indicators', 'algorithm'
    ],
    'commsclaw': [
        'email', 'WhatsApp', 'Telegram', 'message', 'notification', 
        'call', 'social', 'communication', 'contact', 'chat'
    ],
    'coinresearchclaw': [
        'coin research', 'DD', 'due diligence', 'tokenomics', 
        'whitepaper', 'MiCA', 'cryptocurrency', 'token', 'blockchain'
    ],
    'coinsclaw': [
        'coin listing', 'delisting', 'tracked coins', 'v2 upgrade', 
        'CoinsClaw dashboard', 'crypto listings'
    ],
    'paperclipclaw': [
        'Paperclip', 'fleet orchestration', 'agent registry', 
        'company ID', 'coordination', 'orchestration'
    ],
    'devopsoci': [
        'OCI', 'Oracle Cloud', 'compartment', 'ARM instance', 
        'eu-amsterdam', 'cloud infrastructure'
    ],
    'peopleclaw': [
        'HR', 'team', 'culture', 'hiring', 'onboarding', 
        'people', 'recruitment', 'staff'
    ]
}

def classify_content(content: str) -> str:
    """
    Classify content based on keywords to determine the correct agent.
    Returns agent name or 'main' if no clear match.
    """
    content_lower = content.lower()
    
    # Count keyword matches per agent
    agent_scores = {}
    
    for agent, keywords in AGENT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in content_lower:
                score += 1
        agent_scores[agent] = score
    
    # Find agent with highest score
    if agent_scores:
        max_score = max(agent_scores.values())
        if max_score > 0:
            # Get agent with highest score
            best_agents = [agent for agent, score in agent_scores.items() if score == max_score]
            if len(best_agents) == 1:
                return best_agents[0]
            elif len(best_agents) > 1:
                # Multiple agents with same score - return first alphabetically for consistency
                return sorted(best_agents)[0]
    
    # No clear match - assign to main
    return 'main'

def extract_content_from_record(record) -> str:
    """Extract text content from a record's payload"""
    content_fields = ['memory', 'data', 'content', 'text']
    
    content_parts = []
    
    for field in content_fields:
        if field in record.payload:
            value = record.payload[field]
            if isinstance(value, str):
                content_parts.append(value)
            elif isinstance(value, (dict, list)):
                content_parts.append(str(value))
    
    return ' '.join(content_parts)

def main():
    print("🦞 MemClawz Agent Re-attribution Tool")
    print("=" * 50)
    
    # Connect to Qdrant
    client = QdrantClient(host='localhost', port=6333)
    collection = 'yoniclaw_memories'
    
    print(f"Connected to Qdrant collection: {collection}")
    
    # Get all records with missing agent field
    print("Scanning for records with missing agent field...")
    
    offset = None
    missing_agent_records = []
    
    while True:
        result = client.scroll(collection, offset=offset, limit=100)
        records, next_offset = result
        
        if not records:
            break
            
        for record in records:
            if 'agent' not in record.payload or record.payload['agent'] is None:
                missing_agent_records.append(record)
        
        offset = next_offset
        if next_offset is None:
            break
        
        if len(missing_agent_records) % 100 == 0 and missing_agent_records:
            print(f"Found {len(missing_agent_records)} records so far...")
    
    print(f"Found {len(missing_agent_records)} records with missing agent field")
    
    if not missing_agent_records:
        print("No records to process!")
        return
    
    # Process records in batches
    agent_counts = {}
    processed = 0
    
    for i in range(0, len(missing_agent_records), 50):
        batch = missing_agent_records[i:i+50]
        
        for record in batch:
            # Extract content for classification
            content = extract_content_from_record(record)
            
            # Classify content
            assigned_agent = classify_content(content)
            
            # Update the record
            client.set_payload(
                collection_name=collection,
                payload={'agent': assigned_agent},
                points=[record.id]
            )
            
            # Track counts
            agent_counts[assigned_agent] = agent_counts.get(assigned_agent, 0) + 1
            processed += 1
            
            # Log sample classifications
            if processed <= 10:
                content_preview = content[:100].replace('\n', ' ') + '...' if len(content) > 100 else content
                print(f"  Sample {processed}: '{content_preview}' → {assigned_agent}")
        
        # Log progress every 100 records
        if processed % 100 == 0:
            print(f"Processed {processed}/{len(missing_agent_records)} records...")
        
        # Sleep between batches
        time.sleep(1)
    
    print(f"\n✅ Completed processing {processed} records")
    print("\n📊 Re-attribution Summary:")
    print("-" * 30)
    
    for agent, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {agent}: {count}")
    
    # Verify no records still have missing agent field
    print("\n🔍 Verification: Checking for remaining missing agent fields...")
    
    offset = None
    remaining_missing = 0
    
    while True:
        result = client.scroll(collection, offset=offset, limit=1000)
        records, next_offset = result
        
        if not records:
            break
            
        for record in records:
            if 'agent' not in record.payload or record.payload['agent'] is None:
                remaining_missing += 1
        
        offset = next_offset
        if next_offset is None:
            break
    
    print(f"Records still missing agent field: {remaining_missing}")
    
    if remaining_missing == 0:
        print("✅ All records successfully re-attributed!")
    else:
        print(f"⚠️  {remaining_missing} records still need attention")

if __name__ == "__main__":
    main()