"""Upload the app and its pre-computed artefacts to a Hugging Face Space (Docker SDK).

Usage (after running the pipeline locally):
    pip install huggingface_hub
    huggingface-cli login             # stores a write token locally (or: export HF_TOKEN=hf_...)
    python scripts/deploy_hf.py --space <username>/maternal-mortality-xai [--private]

Raw and interim source data are never uploaded; only the derived district-level artefacts
the dashboard needs. Review data-governance requirements before publishing a public Space.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "data/processed/master_district_dataset.csv",
    "data/processed/priority_action_table.csv",
    "data/processed/shap_matrix_class1.parquet",
    "models/best_classifier_pipeline.joblib",
    "models/benchmark_metrics.json",
]
ALLOW = [
    "Dockerfile",
    "requirements.txt",
    "app/**",
    "src/**",
    "config/**",
    "data/processed/**",
    "models/**",
    "reports/figures/**",
]
IGNORE = ["**/__pycache__/**", "**/*.pyc", "data/processed/match_audit.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="Space id, e.g. username/maternal-mortality-xai")
    parser.add_argument("--private", action="store_true", help="Create the Space as private")
    args = parser.parse_args()
    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    if missing:
        sys.exit(f"Run the pipeline first; missing artefacts: {missing}")
    from huggingface_hub import HfApi, get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        sys.exit("No Hugging Face token found. Run `huggingface-cli login` or set HF_TOKEN.")
    api = HfApi(token=token)
    print(f"Authenticated as {api.whoami()['name']}")
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(ROOT / "deploy" / "hf_space_README.md"),
        path_in_repo="README.md",
        repo_id=args.space,
        repo_type="space",
    )
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=args.space,
        repo_type="space",
        allow_patterns=ALLOW,
        ignore_patterns=IGNORE,
        commit_message="Deploy maternal-risk XAI dashboard",
    )
    print(f"Deployed: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
