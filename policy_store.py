"""Validated, local JSON persistence for guardrail policy settings."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any

from analyzer import CATEGORIES

POLICY_PATH = Path(__file__).resolve().parent / "policy_settings.json"
SEVERITY_LEVELS = {"low": 2, "medium": 3, "high": 4, "critical": 5}
RISK_DESCRIPTIONS = {
    "unsafe_content": "Detects actionable instructions that could cause harm or enable wrongdoing.",
    "bias_discrimination": "Detects harmful demographic generalizations and discriminatory recommendations.",
    "confidential_information": "Detects credentials and other sensitive identifiers in the response.",
    "unsupported_claim": "Flags strong factual claims that lack support in the supplied context.",
    "hallucination": "Flags explicit verification claims or narrow contradictions with supplied context.",
    "access_control": "Detects disclosure of protected instructions and apparent access-control bypasses.",
    "toxicity": "Detects direct insults and abusive language aimed at a person.",
    "relevance": "Checks simple requested output formats and instruction-following cues.",
}


def default_policies() -> dict[str, Any]:
    categories: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        categories[category] = {
            "enabled": True,
            "severity": {
                "unsafe_content": "critical", "confidential_information": "critical",
                "access_control": "critical", "bias_discrimination": "high",
                "toxicity": "high", "unsupported_claim": "medium",
                "hallucination": "medium", "relevance": "low",
            }[category],
            "risk_threshold": 0,
            "action": "block" if category in {
                "unsafe_content", "confidential_information", "access_control",
                "bias_discrimination", "toxicity",
            } else "review",
            "description": RISK_DESCRIPTIONS[category],
        }
    return {
        "categories": categories,
        "global": {
            "pass_threshold": 19,
            "fail_threshold": 50,
            "critical_risk_override": True,
            "require_manual_review_uncertain": True,
            "uncertain_confidence_threshold": 0.65,
        },
    }


def validate_policies(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"categories", "global"}:
        raise ValueError("Policy settings must contain categories and global settings.")
    cats, global_settings = value["categories"], value["global"]
    if not isinstance(cats, dict) or set(cats) != set(CATEGORIES):
        raise ValueError("Policy settings must define every supported category exactly once.")
    if not isinstance(global_settings, dict) or set(global_settings) != {
        "pass_threshold", "fail_threshold", "critical_risk_override",
        "require_manual_review_uncertain", "uncertain_confidence_threshold",
    }:
        raise ValueError("Global settings do not match the supported policy schema.")
    clean_categories: dict[str, dict[str, Any]] = {}
    for category, config in cats.items():
        if not isinstance(config, dict) or set(config) != {
            "enabled", "severity", "risk_threshold", "action", "description",
        }:
            raise ValueError(f"Invalid settings for {category}.")
        if not isinstance(config["enabled"], bool):
            raise ValueError(f"{category}.enabled must be true or false.")
        if not isinstance(config["severity"], str) or config["severity"] not in SEVERITY_LEVELS:
            raise ValueError(f"{category}.severity must be low, medium, high, or critical.")
        threshold = config["risk_threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= 100:
            raise ValueError(f"{category}.risk_threshold must be an integer from 0 to 100.")
        if not isinstance(config["action"], str) or config["action"] not in {"allow", "review", "block"}:
            raise ValueError(f"{category}.action must be allow, review, or block.")
        if not isinstance(config["description"], str) or len(config["description"]) > 240:
            raise ValueError(f"{category}.description must be short text.")
        clean_categories[category] = dict(config)
    pass_threshold = global_settings["pass_threshold"]
    fail_threshold = global_settings["fail_threshold"]
    for name, threshold in (("pass_threshold", pass_threshold), ("fail_threshold", fail_threshold)):
        if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= 100:
            raise ValueError(f"global.{name} must be an integer from 0 to 100.")
    if pass_threshold >= fail_threshold:
        raise ValueError("PASS threshold must be lower than FAIL threshold; the gap is REVIEW.")
    for key in ("critical_risk_override", "require_manual_review_uncertain"):
        if not isinstance(global_settings[key], bool):
            raise ValueError(f"global.{key} must be true or false.")
    confidence = global_settings["uncertain_confidence_threshold"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("global.uncertain_confidence_threshold must be between 0 and 1.")
    return {
        "categories": clean_categories,
        "global": {
            "pass_threshold": pass_threshold,
            "fail_threshold": fail_threshold,
            "critical_risk_override": global_settings["critical_risk_override"],
            "require_manual_review_uncertain": global_settings["require_manual_review_uncertain"],
            "uncertain_confidence_threshold": float(confidence),
        },
    }


def load_policies(path: Path = POLICY_PATH) -> dict[str, Any]:
    if not path.exists():
        defaults = default_policies()
        save_policies(defaults, path)
        return defaults
    try:
        return validate_policies(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Saved guardrail policies are invalid. Reset them to defaults before evaluating.") from exc


def save_policies(value: Any, path: Path = POLICY_PATH) -> dict[str, Any]:
    clean = validate_policies(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(clean, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return clean


def reset_policies(path: Path = POLICY_PATH) -> dict[str, Any]:
    return save_policies(default_policies(), path)
