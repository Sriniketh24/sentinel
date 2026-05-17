"""CLI entry point for running the Sentinel pipeline on a single document."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.graph import run_pipeline
from src.config.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run Sentinel analysis pipeline")
    parser.add_argument("--file", type=str, help="Path to document file")
    parser.add_argument("--text", type=str, help="Raw text content to analyze")
    parser.add_argument("--ticker", type=str, default="", help="Company ticker symbol")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    setup_logging()

    if args.file:
        content = Path(args.file).read_text(errors="replace")
    elif args.text:
        content = args.text
    else:
        print("Provide --file or --text")
        sys.exit(1)

    result = run_pipeline(
        document_id="cli_run",
        content=content,
        company_ticker=args.ticker,
    )

    output = result.model_dump()
    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2, default=str))
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
