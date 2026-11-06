from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative_path: str) -> dict[str, Any]:
    with (PROJECT_ROOT / relative_path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_config() -> dict[str, Any]:
    return load_yaml("config/config.yaml")


def feature_domains() -> dict[str, list[str]]:
    return load_yaml("config/feature_domains.yaml")


def scheme_mapping() -> dict[str, dict[str, str]]:
    return load_yaml("config/policy_scheme_mapping.yaml")


def configured_path(config: dict[str, Any], name: str) -> Path:
    return PROJECT_ROOT / config["paths"][name]


def feature_columns() -> list[str]:
    """All *candidate* features declared in feature_domains.yaml (before automatic selection)."""
    return [column for columns in feature_domains().values() for column in columns]


def domain_by_feature() -> dict[str, str]:
    return {column: domain for domain, columns in feature_domains().items() for column in columns}


def zones(config: dict[str, Any] | None = None) -> dict[str, list[str]]:
    return (config or load_config()).get("zones", {})


def pretty(name: str) -> str:
    """Human-readable label for a feature or domain identifier."""
    text = name.replace("hmis_", "HMIS ").replace("_pct", " (%)").replace("_", " ")
    text = text[:1].upper() + text[1:]
    acronyms = {"anc": "ANC", "ifa": "IFA", "oop": "OOP", "bmi": "BMI", "bp": "BP", "mcp": "MCP", "jssk": "JSSK", "wash": "WASH"}
    return " ".join(acronyms.get(word.lower(), word) for word in text.split(" "))
