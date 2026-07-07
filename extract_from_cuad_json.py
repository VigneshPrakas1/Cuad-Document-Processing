#!/usr/bin/env python3
"""
extract_from_cuad_json.py
--------------------------
Extract full contract text from CUADv1.json (SQuAD-style QA format) into
individual .txt files that data_loader.py can read directly.

Usage:
    python extract_from_cuad_json.py --input /path/to/CUADv1.json --output_dir sample_data --n 50
"""
import argparse
import json
import re
from pathlib import Path


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-]", "_", name)
    return name[:150]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CUADv1.json")
    parser.add_argument("--output_dir", default="cuad_contracts_txt", help="Where to write .txt files")
    parser.add_argument("--n", type=int, default=50, help="Max number of contracts to extract")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        cuad = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for entry in cuad["data"]:
        if count >= args.n:
            break

        title = entry["title"]
        contexts = [p.get("context", "") for p in entry.get("paragraphs", [])]
        contexts = [c for c in contexts if c]
        if not contexts:
            continue
        full_text = max(contexts, key=len)

        filename = sanitize_filename(title) + ".txt"
        (out_dir / filename).write_text(full_text, encoding="utf-8")
        count += 1
        print(f"[{count}/{args.n}] wrote {filename} ({len(full_text)} chars)")

    print(f"\nDone. Extracted {count} contracts to {out_dir}/")


if __name__ == "__main__":
    main()
	
