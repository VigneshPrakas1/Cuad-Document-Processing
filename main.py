#!/usr/bin/env python3
"""
main.py
-------
CLI entry point for the CUAD clause-extraction & summarization pipeline.

Usage
-----
    python main.py --data_dir sample_data --output_dir outputs --n_contracts 50

Environment
-----------
Reads GROQ_API_KEY / GROQ_MODEL / USE_MOCK_LLM from a .env file in
the project root (see .env.example) or from the shell environment.
"""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUAD clause extraction & summarization pipeline")
    parser.add_argument(
        "--data_dir",
        default="sample_data",
        help="Directory containing contract .pdf/.txt files (default: sample_data)",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to write results.json / results.csv (default: outputs)",
    )
    parser.add_argument(
        "--n_contracts",
        type=int,
        default=50,
        help="Max number of contracts to process (default: 50, per assignment spec)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()  # picks up .env if present; no-op otherwise

    results = run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_contracts=args.n_contracts,
    )
    print(f"\nProcessed {len(results)} contract(s). Results written to {args.output_dir}/")


if __name__ == "__main__":
    main()
