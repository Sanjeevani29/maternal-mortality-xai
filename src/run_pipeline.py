"""End-to-end pipeline: sources -> merge -> features -> Stage 1 -> Stage 2 -> Stage 3 -> figures/geo."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.configuration import PROJECT_ROOT, configured_path, feature_columns, load_config, zones
from src.data.merge_nfhs_hmis import merge_sources
from src.data.preprocess_impute import engineer_features, select_features
from src.demo_data import write_demo_sources
from src.target.ols_priority_classifier import assert_no_leakage, assign_priority_classes, stage1_model_dict

DEMO_NFHS = PROJECT_ROOT / "data/demo/nfhs5_demo_SYNTHETIC.csv"
DEMO_HMIS = PROJECT_ROOT / "data/demo/hmis_demo_SYNTHETIC.csv"


def prepare_real_sources(config: dict[str, Any], rebuild: bool) -> dict[str, Any]:
    """Build the canonical NFHS and pooled HMIS interim tables from the official raw files."""
    from src.data.hmis_extract import pool_hmis_annual_files
    from src.data.nfhs_extract import build_nfhs_canonical

    nfhs_out, hmis_out = configured_path(config, "nfhs_raw"), configured_path(config, "hmis_raw")
    if rebuild or not nfhs_out.exists():
        print("Building canonical NFHS-5 district table...", flush=True)
        build_nfhs_canonical(configured_path(config, "nfhs_source"), nfhs_out)
    hmis_files = [configured_path(config, f"hmis_{year}") for year in config["real_data"]["hmis_years"]]
    present = [path for path in hmis_files if path.exists()]
    if len(present) < int(config["real_data"]["minimum_hmis_years"]):
        raise FileNotFoundError(f"Need at least {config['real_data']['minimum_hmis_years']} HMIS archive(s).")
    if rebuild or not hmis_out.exists():
        print(f"Extracting HMIS from {len(present)} archive(s) (several minutes)...", flush=True)
        pool_hmis_annual_files(present, hmis_out, progress=True, aliases_path=PROJECT_ROOT / config["matching"]["aliases_path"])
    nfhs = pd.read_csv(nfhs_out)
    if len(nfhs) < int(config["real_data"]["minimum_nfhs_districts"]):
        raise ValueError(f"NFHS extract has {len(nfhs)} districts; expected >= {config['real_data']['minimum_nfhs_districts']}.")
    return {
        "nfhs_source": str(configured_path(config, "nfhs_source").relative_to(PROJECT_ROOT)),
        "hmis_archives_used": [path.name for path in present],
        "hmis_archives_missing": [path.name for path in hmis_files if not path.exists()],
        "hmis_window_note": (
            f"Pooled HMIS window covers {len(present)} of {len(hmis_files)} configured financial years; the "
            "missing archives were not found in data/raw/."
            if len(present) < len(hmis_files)
            else f"HMIS counts pooled over {len(present)} financial years: "
            + ", ".join(path.stem.replace("hmis_", "").replace("_official", "") for path in present)
            + "."
        ),
    }


def run(
    demo: bool = False,
    trials: int | None = None,
    skip_shap: bool = False,
    skip_figures: bool = False,
    rebuild_sources: bool = False,
    skip_validation: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.time()
    config = config or load_config()
    if trials is not None:
        config["modelling"]["optuna_trials"] = trials
    seed = int(config["project"]["random_state"])
    provenance: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "SYNTHETIC DEMO" if demo else "real data",
    }

    if demo:
        write_demo_sources(DEMO_NFHS, DEMO_HMIS, int(config["project"]["demo_rows"]), seed)
        nfhs_path, hmis_path = DEMO_NFHS, DEMO_HMIS
        print("Created SYNTHETIC demo source files. Do not interpret these as real district evidence.")
    else:
        provenance.update(prepare_real_sources(config, rebuild_sources))
        nfhs_path, hmis_path = configured_path(config, "nfhs_raw"), configured_path(config, "hmis_raw")

    matching = config["matching"]
    master = merge_sources(
        nfhs_path,
        hmis_path,
        aliases_path=PROJECT_ROOT / matching["aliases_path"],
        audit_path=configured_path(config, "match_audit"),
        automatic_threshold=int(matching["automatic_match_threshold"]),
        review_threshold=int(matching["review_match_threshold"]),
        minimum_matched_share=float(config["real_data"]["minimum_matched_share"]),
    )
    candidates = feature_columns()
    assert_no_leakage(candidates)
    master = engineer_features(master, candidates, config["preprocessing"])

    minimum_births = float(config["target"].get("minimum_live_births", 0))
    eligible = (
        master["hmis_matched"].astype(bool)
        & (pd.to_numeric(master["hmis_live_births"], errors="coerce") >= minimum_births)
        & master["institutional_births_pct"].notna()
    )
    master["modelling_eligible"] = eligible
    stage1_input = master.loc[eligible].reset_index(drop=True)
    stage1_input["district_key"] = stage1_input["state"].astype(str) + "|" + stage1_input["district"].astype(str)
    modelling, stage1 = assign_priority_classes(stage1_input, config["target"])
    stage1_dict = stage1_model_dict(stage1)

    selected, feature_audit = select_features(modelling, candidates, config["preprocessing"])
    assert_no_leakage(selected)
    for name in ("feature_audit", "stage1_model", "provenance"):
        configured_path(config, name).parent.mkdir(parents=True, exist_ok=True)
    feature_audit.to_csv(configured_path(config, "feature_audit"), index=False)
    print(f"Districts: {len(master)} total, {int(eligible.sum())} eligible. Features kept: {len(selected)}/{len(candidates)}")

    # Persist the full district frame (ineligible districts keep empty Stage-1 fields).
    stage1_columns = [c for c in modelling.columns if c not in master.columns]
    full = master.merge(modelling[["state", "district", *stage1_columns]], on=["state", "district"], how="left")
    master_path = configured_path(config, "master_dataset")
    master_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(master_path, index=False)
    joblib.dump({**stage1_dict, "feature_columns": selected}, configured_path(config, "stage1_model"))

    from src.models.train_evaluate import train_and_evaluate

    model, metrics = train_and_evaluate(
        modelling,
        selected,
        config["modelling"],
        seed,
        configured_path(config, "model"),
        configured_path(config, "metrics"),
        calibrated_path=configured_path(config, "calibrated_model"),
        zones=zones(config),
        n_neighbors=int(config["preprocessing"]["knn_neighbors"]),
    )
    print(f"Selected model: {metrics['selected_model']}")

    if not skip_shap:
        from src.explain.shap_profiler import generate_shap_artifacts

        calibrated = joblib.load(configured_path(config, "calibrated_model"))
        _, actions = generate_shap_artifacts(
            modelling,
            model,
            selected,
            configured_path(config, "shap_matrix"),
            configured_path(config, "priority_table"),
            domain_share_path=configured_path(config, "domain_shares"),
            calibrated_model=calibrated,
            low_access_audit_threshold=float(config["target"]["low_access_audit_threshold"]),
        )
        print(f"Wrote {len(actions)} district action profiles.")

    figures: list[str] = []
    if not skip_figures:
        from src.reports.eda import generate_eda_figures

        figure_dir = configured_path(config, "figures")
        figures += [str(p.relative_to(PROJECT_ROOT)) for p in generate_eda_figures(modelling, selected, stage1_dict, figure_dir)]
        if not skip_shap:
            from src.explain.shap_profiler import save_shap_figures

            figures += [str(p.relative_to(PROJECT_ROOT)) for p in save_shap_figures(model, modelling, selected, figure_dir)]

    if not skip_validation:
        from src.models.validate import run_validation

        print("Validating: random vs state-grouped CV and Stage-1 label sensitivity...", flush=True)
        validation = run_validation(
            modelling,
            stage1_input,
            selected,
            metrics,
            config,
            configured_path(config, "validation_report"),
            None if skip_figures else configured_path(config, "figures"),
        )
        provenance["validation"] = {
            scheme: validation["cv"][scheme]["pooled"] for scheme in ("random_stratified", "state_grouped")
        }
        figures += [f"reports/figures/{name}" for name in validation.get("figures", [])]

    geo_path = configured_path(config, "geojson_simplified")
    if demo:
        geo_path.unlink(missing_ok=True)  # synthetic districts have no geometry
    else:
        from src.geo.build_geojson import build_simplified_geojson

        provenance["geojson"] = build_simplified_geojson(
            configured_path(config, "district_geojson"),
            master,
            geo_path,
            configured_path(config, "geo_match_audit"),
            PROJECT_ROOT / matching["aliases_path"],
        )

    selected_metrics = metrics["candidate_models"][metrics["selected_model"]]
    provenance.update(
        {
            "districts_total": int(len(master)),
            "districts_hmis_matched": int(master["hmis_matched"].sum()),
            "districts_modelled": int(eligible.sum()),
            "minimum_live_births": minimum_births,
            "features_candidate": len(candidates),
            "features_selected": selected,
            "class_distribution": metrics["class_distribution"],
            "stage1": stage1_dict,
            "selected_model": metrics["selected_model"],
            "holdout_class_1_recall": selected_metrics["holdout"]["class_1_recall"],
            "holdout_macro_f1": selected_metrics["holdout"]["macro_f1"],
            "task": metrics["task"],
            "decision_threshold": metrics["decision_threshold"],
            "holdout_class_1_recall_at_threshold": selected_metrics.get("holdout_at_threshold", {}).get("class_1_recall"),
            "holdout_class_1_precision_at_threshold": selected_metrics.get("holdout_at_threshold", {}).get("class_1_precision"),
            "figures": figures,
            "runtime_seconds": round(time.time() - started, 1),
        }
    )
    configured_path(config, "provenance").write_text(json.dumps(provenance, indent=2, default=float), encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the maternal-risk research pipeline.")
    parser.add_argument("--demo", action="store_true", help="Use SYNTHETIC demo inputs instead of the official data.")
    parser.add_argument("--trials", type=int, default=None, help="Override Optuna trials per model (0 = defaults).")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP artefacts.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip EDA/SHAP figures.")
    parser.add_argument("--rebuild-sources", action="store_true", help="Re-extract NFHS/HMIS interim tables.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip state-grouped CV and label sensitivity.")
    args = parser.parse_args()
    summary = run(args.demo, args.trials, args.skip_shap, args.skip_figures, args.rebuild_sources, args.skip_validation)
    print(json.dumps({key: summary[key] for key in summary if key not in {"features_selected", "stage1"}}, indent=2, default=float))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
