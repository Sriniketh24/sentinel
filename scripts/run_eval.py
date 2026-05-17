"""CLI entry point for running the eval harness."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.logging import setup_logging
from src.eval.runner import EvalRunner


def main():
    setup_logging()
    runner = EvalRunner()
    metrics = runner.run()
    print(json.dumps(metrics.summary(), indent=2))


if __name__ == "__main__":
    main()
