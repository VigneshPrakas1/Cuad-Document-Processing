#!/usr/bin/env python3
"""
search.py
---------
Bonus CLI: semantic search over clauses already extracted by main.py.

Usage
-----
    python main.py --data_dir sample_data --output_dir outputs   # run once first
    python search.py --results outputs/results.json --query "termination without cause"
"""

from __future__ import annotations

import argparse
import json

from src.embeddings_search import ClauseSearchIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic search over extracted contract clauses")
    parser.add_argument("--results", default="outputs/results.json", help="Path to results.json from main.py")
    parser.add_argument("--query", required=True, help="Natural-language search query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)

    index = ClauseSearchIndex()
    index.build(results)
    hits = index.search(args.query, top_k=args.top_k)

    if not hits:
        print("No clause excerpts found to search over (did you run main.py first?).")
        return

    print(f'Top {len(hits)} matches for: "{args.query}"\n')
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. [{hit['score']:.3f}] {hit['contract_id']} :: {hit['clause_type']}")
        print(f"   {hit['excerpt']}\n")


if __name__ == "__main__":
    main()
