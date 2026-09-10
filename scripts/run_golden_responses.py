from __future__ import annotations

import argparse

from hiver_support.evaluation.responses import run_response_evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=48)
    args = parser.parse_args()
    result = run_response_evaluation(args.size)
    print(f"Completed {len(result['records'])} response evaluations")
