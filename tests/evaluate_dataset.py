"""Run the labeled benchmark from the command line and print computed metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import run_benchmark  # noqa: E402


def show_rate(name: str, metric: dict[str, object]) -> str:
    value = metric["value"]
    return f"{name}: {value:.3f}" if isinstance(value, (int, float)) else f"{name}: Undefined (denominator is zero)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic chatbot guardrail benchmark.")
    parser.add_argument("--review-handling", choices=("exclude", "positive", "negative"), default="exclude",
                        help="How binary PASS/FAIL metrics treat REVIEW outputs (default: exclude).")
    args = parser.parse_args()
    report = run_benchmark(args.review_handling)
    binary = report["decision_binary_metrics"]
    print(f"Labeled examples: {report['dataset_size']} ({report['dataset_classes']})")
    print(f"Benchmark run ID: {report['benchmark_run_id']} · Dataset SHA-256: {report['dataset_sha256']}")
    print(f"Binary positive class: {binary['positive_class']}; REVIEW handling: {binary['review_handling']}")
    print(f"Evaluated: {binary['evaluated_examples']}; excluded for REVIEW: {binary['excluded_for_review']}")
    print(f"TP={binary['true_positives']} TN={binary['true_negatives']} FP={binary['false_positives']} FN={binary['false_negatives']}")
    for key in ("accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate"):
        print(show_rate(key, binary[key]))
    print("\nPer-category one-vs-rest metrics (N is the evaluated example count):")
    for category, metric in report["category_metrics"].items():
        print(f"{category:28} N={metric['evaluated_examples']:>3} TP={metric['true_positives']:>2} TN={metric['true_negatives']:>2} FP={metric['false_positives']:>2} FN={metric['false_negatives']:>2} "
              f"P={metric['precision']['value']} R={metric['recall']['value']} F1={metric['f1']['value']}")
    print("\nDecision confusion matrix (expected rows, actual columns):")
    for expected, columns in report["decision_confusion_matrix"].items():
        print(f"{expected:>7}: " + "  ".join(f"{actual}={columns[actual]}" for actual in ("PASS", "REVIEW", "FAIL")))
    print(f"\nCategory false positives/negatives: {len(report['category_errors'])}")
    print(f"Decision/category/severity mismatches: {len(report['misclassified_examples'])}")
    for limitation in report["limitations"]:
        print(f"Note: {limitation}")


if __name__ == "__main__":
    main()
