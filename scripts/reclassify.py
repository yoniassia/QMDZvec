#!/usr/bin/env python3
"""One-time job: classify all existing Qdrant vectors that lack a 'type' field."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from memclawz.classifier import classify_memory
from memclawz.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def reclassify_all():
    offset = None
    total = 0
    classified = 0
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        for point in results:
            total += 1
            payload = point.payload or {}
            if not payload.get("type"):
                text = payload.get("memory", payload.get("data", ""))
                if text:
                    mem_type = classify_memory(text)
                    client.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={"type": mem_type},
                        points=[point.id],
                    )
                    classified += 1
                    if classified % 100 == 0:
                        print(f"  Classified {classified} so far...")

        if offset is None:
            break

    print(f"Done! Scanned {total} points, classified {classified} untyped memories.")


if __name__ == "__main__":
    reclassify_all()
