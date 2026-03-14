#!/usr/bin/env python3
"""
Redistribute "main" memories to correct domain agents in MemClawz Qdrant
"""

from qdrant_client import QdrantClient
import json
import time
import re
from typing import Dict, List, Optional, Tuple

# Enhanced agent keyword classification rules (be AGGRESSIVE in matching)
AGENT_KEYWORDS = {
    'infraclaw': [
        'server', 'deploy', 'DNS', 'Caddy', 'Docker', 'SSH', 'firewall', 
        'Hetzner', 'systemd', 'nginx', 'SSL', 'port', 'certificate', 
        'reverse proxy', 'VPS', 'Ubuntu', 'PID', 'gateway', 'restart', 
        'service', 'daemon', 'uptime', 'Coolify', 'CI/CD', 'backup', 
        'cron job', 'fleet sync', 'Qdrant crash', 'process kill', 
        'config file', 'security hardening', 'iptables', 'ufw', 
        'disk space', 'monitoring', 'healthcheck', 'node setup', 'PM2', 'socat'
    ],
    
    'tradeclaw': [
        'trade', 'portfolio', 'market', 'stock', 'ETF', 'DeFi', 'ClawMM', 
        'Hyperliquid', 'brief', 'holdings', 'BTC', 'crypto price', 'Bitcoin', 
        'Ethereum', 'buy', 'sell', 'position', 'P&L', 'rebalance', 'yield', 
        'APY', 'hedge', 'leverage', 'liquidation', 'margin', 'DEX', 'CEX', 
        'Uniswap', 'token price', 'market cap', 'daily brief', 'macro', 
        'earnings', 'S&P', 'NASDAQ', 'Dow', 'watchlist', 'conviction', 
        'alpha', 'signal strategy', 'backtest result'
    ],
    
    'appsclaw': [
        'app store', 'swagger', 'plugin', 'eToro app', 'liquidation map', 
        'dashboard app', 'ClawX', 'SuperApp', 'banking pillar', 'app review', 
        'frontend component', 'UI mockup', 'Vercel', 'Next.js app', 
        'React component', 'widget', 'iframe', 'app listing', 'plugin marketplace'
    ],
    
    'tradingdataclaw': [
        'data module', 'AlphaEar', 'DataScout', 'quant', 'signal', 
        'backtest', 'VectorBT', 'MCP data', 'data pipeline', 'scraper', 
        'API endpoint data', 'financial dataset', 'OHLCV', 'candlestick', 
        'indicator', 'RSI', 'MACD', 'moving average', 'data feed', 
        'websocket stream', 'market data provider'
    ],
    
    'commsclaw': [
        'email', 'WhatsApp config', 'Telegram config', 'message routing', 
        'notification', 'social media post', 'Twitter/X post', 'LinkedIn', 
        'call', 'SMS', 'channel configuration', 'bot token', 'group policy', 
        'DM policy', 'allowFrom', 'requireMention'
    ],
    
    'coinresearchclaw': [
        'coin research', 'DD', 'due diligence', 'tokenomics', 'whitepaper', 
        'MiCA', 'token analysis', 'blockchain project', 'crypto fundamentals', 
        'market thesis', 'research memo', 'risk assessment crypto'
    ],
    
    'coinsclaw': [
        'coin listing', 'delisting', 'tracked coins', 'v2 upgrade', 
        'CoinsClaw dashboard', 'coin database', '191 coins', 'wave', 
        'listing criteria', 'exchange listing'
    ],
    
    'paperclipclaw': [
        'Paperclip', 'fleet orchestration', 'agent registry', 'company ID', 
        'org chart', 'agent registration', 'fleet dashboard', 'master.clawz.org', 
        'agent coordination', 'Paperclip API', 'issue create'
    ],
    
    'devopsoci': [
        'OCI', 'Oracle Cloud', 'compartment', 'ARM instance', 'eu-amsterdam', 
        'claw-fleet-ams', 'Always Free', 'VCN', 'subnet', 'JackBlackClaw on OCI', 
        'DevOpsClaw', '141.144.203.233'
    ],
    
    'peopleclaw': [
        'HR', 'team', 'culture', 'hiring', 'onboarding', 'employee', 
        'org structure', 'people management', 'team directory', 'roles'
    ],
    
    'cmoclaw': [
        'marketing', 'campaign', 'brand', 'creative', 'ad', 'content strategy', 
        'social media strategy', 'PR', 'press release', 'CMO', 'growth', 
        'SEO', 'influencer', 'partnership marketing'
    ],
    
    'moneyclawx': [
        'MoneyClaw', 'Tori', 'tenant', 'moltbook', 'real estate', 
        'overnight build', 'consumer app', 'onboarding wizard', 
        'gamification', 'AgentX'
    ],
    
    'quantclaw': [
        'quant strategy', 'QuantClaw', 'Freqtrade', 'algorithmic trading', 
        'backtesting framework', 'signal generation', 'risk model', 
        'portfolio optimization', 'factor model'
    ],
    
    'qaclaw': [
        'QA', 'testing', 'Applause', 'test case', 'bug report', 'regression', 
        'Cypress', 'Playwright', 'test automation', 'ClawQA', 'test results', 
        'browser testing'
    ]
}

def classify_memory_content(memory_text: str) -> Tuple[str, int]:
    """
    Classify memory content based on keywords to determine the correct agent.
    Returns (agent_name, score) where score is the number of keyword matches.
    Only re-attributes if score >= 2 to avoid false positives.
    """
    memory_lower = memory_text.lower()
    
    # Count keyword matches per agent
    agent_scores = {}
    
    for agent, keywords in AGENT_KEYWORDS.items():
        score = 0
        matched_keywords = []
        
        for keyword in keywords:
            if keyword.lower() in memory_lower:
                score += 1
                matched_keywords.append(keyword)
        
        if score > 0:
            agent_scores[agent] = (score, matched_keywords)
    
    # Find agent with highest score
    if agent_scores:
        best_agent = max(agent_scores.keys(), key=lambda x: agent_scores[x][0])
        best_score, matched_keywords = agent_scores[best_agent]
        
        # Only re-attribute if score >= 2
        if best_score >= 2:
            return best_agent, best_score
    
    # Keep as main if no strong match
    return 'main', 0

def main():
    print("🦞 MemClawz Main Memory Redistribution Tool")
    print("=" * 60)
    
    # Connect to Qdrant
    client = QdrantClient(host='localhost', port=6333)
    collection = 'yoniclaw_memories'
    
    print(f"Connected to Qdrant collection: {collection}")
    
    # Get all records with agent_id = "main"
    print("Fetching all records with agent_id='main'...")
    
    main_records = []
    offset = None
    fetched_count = 0
    
    while True:
        # Use scroll with filter for agent_id = "main"
        result = client.scroll(
            collection, 
            offset=offset, 
            limit=100,
            scroll_filter={
                "must": [
                    {"key": "agent_id", "match": {"value": "main"}}
                ]
            }
        )
        records, next_offset = result
        
        if not records:
            break
            
        main_records.extend(records)
        fetched_count += len(records)
        
        if fetched_count % 500 == 0:
            print(f"  Fetched {fetched_count} main records so far...")
        
        offset = next_offset
        if next_offset is None:
            break
    
    print(f"Found {len(main_records)} records with agent_id='main'")
    
    if not main_records:
        print("No 'main' records to process!")
        return
    
    # Process records in batches
    agent_move_counts = {}
    processed = 0
    moved = 0
    
    print(f"\nProcessing {len(main_records)} records...")
    print("Starting redistribution based on keyword matching...")
    
    for i in range(0, len(main_records), 100):
        batch = main_records[i:i+100]
        
        for record in batch:
            # Extract memory content
            memory_text = record.payload.get('memory', '')
            
            if not memory_text:
                # Try other content fields as fallback
                memory_text = record.payload.get('content', '') or record.payload.get('data', '') or str(record.payload)
            
            # Classify content
            new_agent, score = classify_memory_content(memory_text)
            
            # Only update if we're moving to a different agent
            if new_agent != 'main':
                # Update the record with new agent_id
                client.set_payload(
                    collection_name=collection,
                    payload={'agent_id': new_agent},
                    points=[record.id]
                )
                
                # Track counts
                agent_move_counts[new_agent] = agent_move_counts.get(new_agent, 0) + 1
                moved += 1
                
                # Log first few classifications as examples
                if moved <= 10:
                    memory_preview = memory_text[:150].replace('\n', ' ') + '...' if len(memory_text) > 150 else memory_text
                    print(f"  Sample {moved}: (score:{score}) '{memory_preview}' → {new_agent}")
            
            processed += 1
            
            # Log progress every 500 records
            if processed % 500 == 0:
                print(f"  Progress: {processed}/{len(main_records)} processed, {moved} moved")
        
        # Sleep between batches to be gentle on the system
        time.sleep(0.5)
    
    print(f"\n✅ Completed processing {processed} records")
    print(f"📊 Moved {moved} records from 'main' to domain agents")
    print(f"📊 {processed - moved} records remain as 'main'")
    
    print("\n📋 Redistribution Summary:")
    print("-" * 40)
    
    if agent_move_counts:
        for agent, count in sorted(agent_move_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {agent}: {count} records moved")
    else:
        print("  No records were moved (all remained as 'main')")
    
    # Final verification - check current distribution
    print("\n🔍 Final Verification: Current agent_id distribution...")
    
    final_agent_counts = {}
    offset = None
    
    while True:
        result = client.scroll(collection, offset=offset, limit=1000)
        records, next_offset = result
        
        if not records:
            break
            
        for record in records:
            agent_id = record.payload.get('agent_id', 'unknown')
            final_agent_counts[agent_id] = final_agent_counts.get(agent_id, 0) + 1
        
        offset = next_offset
        if next_offset is None:
            break
    
    print("Current distribution after redistribution:")
    for agent_id, count in sorted(final_agent_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {agent_id}: {count}")

if __name__ == "__main__":
    main()