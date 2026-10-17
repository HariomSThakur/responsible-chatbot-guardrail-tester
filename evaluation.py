"""Reproducible local adversarial and labeled benchmark runs."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from analyzer import CATEGORIES, analyze
from policy_store import load_policies, validate_policies

ROOT = Path(__file__).resolve().parent
ADVERSARIAL_PATH = ROOT / "tests" / "adversarial_cases.json"
BENCHMARK_PATH = ROOT / "tests" / "benchmark_dataset.json"
DECISIONS = ("PASS", "REVIEW", "FAIL")
REVIEW_HANDLING = {"exclude", "positive", "negative"}


def load_dataset(path: Path) -> list[dict[str, Any]]:
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read evaluation dataset ({path.name}).") from exc
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError(f"Evaluation dataset {path.name} must be a list of objects.")
    ids: set[str] = set()
    for item in items:
        case_id = item.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError(f"Evaluation dataset {path.name} has an empty or duplicate ID.")
        ids.add(case_id)
        for field in ("prompt", "response", "reference_context"):
            if not isinstance(item.get(field, ""), str):
                raise ValueError(f"Evaluation case {case_id} has invalid {field} text.")
    return items


def _actual_categories(result: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(
        finding["category"] for finding in result["findings"]
        if finding.get("policy_status") in {"active", "mandatory_protection"}
    ))


def run_adversarial(test_id: str | None = None, policies: dict[str, Any] | None = None) -> dict[str, Any]:
    suite = load_dataset(ADVERSARIAL_PATH)
    if policies is None:
        policies = load_policies()
    else:
        policies = validate_policies(policies)
    if test_id is not None:
        suite = [case for case in suite if case["id"] == test_id]
        if not suite:
            raise ValueError("Unknown adversarial test ID.")
    results: list[dict[str, Any]] = []
    by_category: dict[str, dict[str, int]] = {}
    for case in suite:
        expected = case.get("expected_outcome")
        expected_category = case.get("expected_risk_category")
        if (expected not in DECISIONS
                or (expected_category is not None and
                    (not isinstance(expected_category, str) or expected_category not in CATEGORIES))):
            raise ValueError(f"Adversarial case {case['id']} has invalid expected labels.")
        category = case.get("category", "Uncategorized")
        counts = by_category.setdefault(category, {"total": 0, "passed": 0, "failed": 0, "execution_errors": 0})
        counts["total"] += 1
        try:
            actual = analyze(case, policies, use_llm=False)
            actual_categories = _actual_categories(actual)
            expected_category = case.get("expected_risk_category")
            category_matches = expected_category is None or expected_category in actual_categories
            decision_matches = actual["decision"] == case["expected_outcome"]
            passed = category_matches and decision_matches
            explanation = (
                f"Expected {case['expected_outcome']} and {expected_category or 'no active risk category'}; "
                f"got {actual['decision']} and {', '.join(actual_categories) if actual_categories else 'no active risk category'}."
            )
            results.append({
                "id": case["id"], "category": category,
                "expected": {"outcome": case["expected_outcome"], "risk_category": expected_category},
                "actual": {"outcome": actual["decision"], "risk_categories": actual_categories,
                           "risk_score": actual["overall_risk_score"]},
                "execution_status": "success", "test_status": "PASS" if passed else "FAIL",
                "explanation": explanation,
            })
            counts["passed" if passed else "failed"] += 1
        except Exception as exc:
            results.append({
                "id": case["id"], "category": category,
                "expected": {"outcome": case.get("expected_outcome"), "risk_category": case.get("expected_risk_category")},
                "actual": None, "execution_status": "error", "test_status": "NOT RUN",
                "explanation": f"Test execution failed ({type(exc).__name__}); this is not a chatbot safety failure.",
            })
            counts["execution_errors"] += 1
    return {
        "run_type": "single" if test_id else "suite",
        "total_tests": len(results),
        "tests_passed": sum(item["test_status"] == "PASS" for item in results),
        "tests_failed": sum(item["test_status"] == "FAIL" for item in results),
        "execution_errors": sum(item["execution_status"] == "error" for item in results),
        "by_category": by_category,
        "results": results,
    }


def _safe_rate(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator == 0:
        return {"value": None, "defined": False, "note": "Undefined: denominator is zero."}
    return {"value": round(numerator / denominator, 4), "defined": True, "note": None}


def _metrics(tp: int, tn: int, fp: int, fn: int, evaluated: int, excluded: int = 0) -> dict[str, Any]:
    precision = _safe_rate(tp, tp + fp)
    recall = _safe_rate(tp, tp + fn)
    f1_denominator = 2 * tp + fp + fn
    f1 = _safe_rate(2 * tp, f1_denominator)
    return {
        "true_positives": tp, "true_negatives": tn,
        "false_positives": fp, "false_negatives": fn,
        "evaluated_examples": evaluated, "excluded_for_review": excluded,
        "accuracy": _safe_rate(tp + tn, evaluated),
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": _safe_rate(fp, fp + tn),
        "false_negative_rate": _safe_rate(fn, fn + tp),
    }


def _decision_confusion(examples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {expected: {actual: 0 for actual in DECISIONS} for expected in DECISIONS}
    for example in examples:
        matrix[example["expected"]["decision"]][example["actual"]["decision"]] += 1
    return matrix


def _decision_binary(examples: list[dict[str, Any]], review_handling: str) -> dict[str, Any]:
    tp = tn = fp = fn = excluded = evaluated = 0
    for example in examples:
        expected = example["expected"]["decision"]
        actual = example["actual"]["decision"]
        if review_handling == "exclude" and "REVIEW" in (expected, actual):
            excluded += 1
            continue
        positive_label = "FAIL"
        if review_handling == "positive":
            expected_positive = expected in {"FAIL", "REVIEW"}
            actual_positive = actual in {"FAIL", "REVIEW"}
        elif review_handling == "negative":
            expected_positive = expected == positive_label
            actual_positive = actual == positive_label
        else:
            expected_positive = expected == positive_label
            actual_positive = actual == positive_label
        evaluated += 1
        if expected_positive and actual_positive: tp += 1
        elif not expected_positive and not actual_positive: tn += 1
        elif actual_positive: fp += 1
        else: fn += 1
    return {"positive_class": "FAIL", "review_handling": review_handling,
            **_metrics(tp, tn, fp, fn, evaluated, excluded)}


def run_benchmark(review_handling: str = "exclude", policies: dict[str, Any] | None = None) -> dict[str, Any]:
    if review_handling not in REVIEW_HANDLING:
        raise ValueError("review_handling must be exclude, positive, or negative.")
    if policies is None:
        policies = load_policies()
    else:
        policies = validate_policies(policies)
    data = load_dataset(BENCHMARK_PATH)
    expected_categories = set(CATEGORIES)
    examples: list[dict[str, Any]] = []
    category_counts = {category: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for category in CATEGORIES}
    category_errors: list[dict[str, Any]] = []
    for case in data:
        expected_decision = case.get("expected_decision")
        expected_category = case.get("expected_category")
        expected_class = case.get("expected_class")
        expected_severity = case.get("expected_severity")
        if (expected_decision not in DECISIONS
                or (expected_category is not None and
                    (not isinstance(expected_category, str) or expected_category not in expected_categories))
                or not isinstance(expected_class, str) or expected_class not in {"safe", "unsafe", "borderline"}
                or isinstance(expected_severity, bool) or not isinstance(expected_severity, int)
                or not 0 <= expected_severity <= 5):
            raise ValueError(f"Benchmark case {case['id']} has invalid labels.")
        actual = analyze(case, policies, use_llm=False)
        actual_categories = _actual_categories(actual)
        expected_categories_for_case = {expected_category} if expected_category else set()
        severity_by_category = {
            category: max((finding["severity"] for finding in actual["findings"]
                           if finding["category"] == category
                           and finding.get("policy_status") in {"active", "mandatory_protection"}), default=0)
            for category in CATEGORIES
        }
        predicted_severity = severity_by_category.get(expected_category, 0) if expected_category else max(severity_by_category.values(), default=0)
        example = {
            "id": case["id"], "class_label": case.get("expected_class", "unlabeled"),
            "prompt": case["prompt"], "response": case["response"],
            "expected": {"decision": expected_decision, "category": expected_category,
                         "severity": case.get("expected_severity", 0)},
            "actual": {"decision": actual["decision"], "categories": actual_categories,
                       "severity": predicted_severity, "severity_by_category": severity_by_category},
            "category_findings": {category: category in actual_categories for category in CATEGORIES},
        }
        example["decision_match"] = actual["decision"] == expected_decision
        example["category_match"] = set(actual_categories) == expected_categories_for_case
        example["severity_match"] = predicted_severity == case.get("expected_severity", 0)
        example["evaluation_status"] = "MATCH" if (
            example["decision_match"] and example["category_match"] and example["severity_match"]
        ) else "MISMATCH"
        examples.append(example)
        for category in CATEGORIES:
            expected_positive = category in expected_categories_for_case
            predicted_positive = category in actual_categories
            counts = category_counts[category]
            if expected_positive and predicted_positive: counts["tp"] += 1
            elif not expected_positive and not predicted_positive: counts["tn"] += 1
            elif predicted_positive: counts["fp"] += 1
            else: counts["fn"] += 1
            if expected_positive != predicted_positive:
                category_errors.append({
                    "id": case["id"], "category": category,
                    "error_type": "false_positive" if predicted_positive else "false_negative",
                    "expected_category": expected_category,
                    "predicted_categories": actual_categories,
                    "expected_class": case.get("expected_class", "unlabeled"),
                    "prompt": case["prompt"], "response": case["response"],
                })
    category_metrics = {}
    for category, counts in category_counts.items():
        category_metrics[category] = _metrics(
            counts["tp"], counts["tn"], counts["fp"], counts["fn"], len(data)
        )
    misclassified = [example for example in examples if example["evaluation_status"] == "MISMATCH"]
    canonical_dataset = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    canonical_run = json.dumps({"dataset": data, "policies": policies, "review_handling": review_handling},
                               ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "dataset_size": len(data),
        "dataset_sha256": hashlib.sha256(canonical_dataset).hexdigest(),
        "benchmark_run_id": hashlib.sha256(canonical_run).hexdigest()[:16],
        "policy_snapshot": policies,
        "dataset_classes": {label: sum(item.get("expected_class") == label for item in data)
                            for label in ("safe", "unsafe", "borderline")},
        "deterministic": True,
        "review_handling": review_handling,
        "review_handling_definition": {
            "exclude": "Exclude a record from the binary PASS/FAIL decision metrics when either expected or actual decision is REVIEW.",
            "positive": "Treat REVIEW as positive (FAIL) in the binary PASS/FAIL decision metrics.",
            "negative": "Treat REVIEW as negative (PASS) in the binary PASS/FAIL decision metrics.",
        }[review_handling],
        "decision_confusion_matrix": _decision_confusion(examples),
        "decision_binary_metrics": _decision_binary(examples, review_handling),
        "category_metrics": category_metrics,
        "category_errors": category_errors,
        "misclassified_examples": misclassified,
        "examples": examples,
        "limitations": [
            "This benchmark uses a small synthetic dataset and is not evidence of production-level reliability.",
            "Category metrics use one-vs-rest labels: the expected category is positive; other categories are negative.",
            "Binary PASS/FAIL decision metrics use the explicitly selected REVIEW handling policy. The three-class confusion matrix keeps REVIEW separate.",
        ],
    }
