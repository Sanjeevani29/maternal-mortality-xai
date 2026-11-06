from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process


REQUIRED_NFHS_COLUMNS = {"district", "state", "institutional_births_pct"}
REQUIRED_HMIS_COLUMNS = {
    "district",
    "state",
    "hmis_maternal_deaths",
    "hmis_live_births",
    "hmis_institutional_deliveries",
}
COMPLICATION_COMPONENTS = {
    "hmis_pph_cases",
    "hmis_eclampsia_cases",
    "hmis_sepsis_cases",
    "hmis_ruptured_uterus_cases",
}


def _clean(value: object) -> str:
    text = str(value if value is not None else "").lower().strip()
    text = text.replace("&", " and ")
    # "urban"/"rural" are deliberately kept: they distinguish real district pairs
    # (Bangalore vs Bangalore Rural, Warangal Urban vs Warangal Rural).
    text = re.sub(r"\bdistrict\b", "", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_aliases(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "districts" not in raw:  # legacy flat format
        raw = {"states": {}, "districts": raw}
    return {
        "states": {_clean(key): _clean(value) for key, value in raw.get("states", {}).items()},
        "districts": {_clean(key): _clean(value) for key, value in raw.get("districts", {}).items()},
        "hmis_state_overrides": {
            _clean(key): _clean(value) for key, value in raw.get("hmis_state_overrides", {}).items()
        },
        "hmis_aggregate": {
            _clean(state): {_clean(parent): [_clean(child) for child in children] for parent, children in groups.items()}
            for state, groups in raw.get("hmis_aggregate", {}).items()
        },
    }


def aggregate_split_districts(hmis: pd.DataFrame, aliases: dict[str, dict]) -> pd.DataFrame:
    """Sum HMIS counts of districts carved out after NFHS-5 back into the NFHS parent district.

    HMIS fields are raw counts, so summation is exact. Only explicitly configured groups are combined.
    """
    groups = aliases["hmis_aggregate"]
    if not groups:
        return hmis
    hmis = hmis.copy()
    count_columns = [c for c in hmis.columns if c.startswith("hmis_") and c != "hmis_years_pooled"]
    drop_rows: list[int] = []
    new_rows: list[dict[str, object]] = []
    for state, parents in groups.items():
        in_state = hmis["_state_norm"] == state
        for parent, children in parents.items():
            members = hmis.index[in_state & hmis["_district_norm"].isin(children)].tolist()
            if len(members) < 2:
                continue
            combined = hmis.loc[members].iloc[0].to_dict()
            for column in count_columns:
                combined[column] = hmis.loc[members, column].sum(min_count=1)
            combined["district"] = " + ".join(hmis.loc[members, "district"].astype(str))
            combined["_district_norm"] = parent
            new_rows.append(combined)
            drop_rows.extend(members)
    if not new_rows:
        return hmis
    return pd.concat([hmis.drop(index=drop_rows), pd.DataFrame(new_rows)], ignore_index=True)


def normalise_state(value: object, aliases: dict[str, dict]) -> str:
    text = _clean(value)
    return aliases["states"].get(text, text)


def normalise_district(value: object, aliases: dict[str, dict]) -> str:
    text = _clean(value)
    return aliases["districts"].get(text, text)


def _validate_columns(frame: pd.DataFrame, required: set[str], source_name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {', '.join(missing)}")


def merge_sources(
    nfhs_path: Path,
    hmis_path: Path,
    aliases_path: Path,
    audit_path: Path,
    automatic_threshold: int = 90,
    review_threshold: int = 75,
    minimum_matched_share: float = 0.85,
) -> pd.DataFrame:
    """Join NFHS and HMIS within state and keep a row-level reconciliation audit.

    Every NFHS district is retained. Districts without a confident HMIS match get empty HMIS fields
    (and are therefore excluded from Stage 1/2 later); they are never force-matched. The merge fails
    only if the matched share drops below ``minimum_matched_share``.
    """
    nfhs = pd.read_csv(nfhs_path)
    hmis = pd.read_csv(hmis_path)
    _validate_columns(nfhs, REQUIRED_NFHS_COLUMNS, "NFHS input")
    _validate_columns(hmis, REQUIRED_HMIS_COLUMNS, "HMIS input")
    if "hmis_obstetric_complications" not in hmis and not COMPLICATION_COMPONENTS.issubset(hmis.columns):
        raise ValueError("HMIS input needs hmis_obstetric_complications or the four complication components.")
    nfhs = nfhs.loc[~nfhs["district"].astype(str).str.lower().str.contains("data not available")].copy()
    if nfhs.duplicated(["state", "district"]).any() or hmis.duplicated(["state", "district"]).any():
        raise ValueError("Duplicate state/district records found; reconcile source records before merging.")

    aliases = load_aliases(aliases_path)
    nfhs = nfhs.reset_index(drop=True)
    hmis = hmis.reset_index(drop=True)
    for frame in (nfhs, hmis):
        frame["_district_norm"] = frame["district"].map(lambda value: normalise_district(value, aliases))
        frame["_state_norm"] = frame["state"].map(lambda value: normalise_state(value, aliases))
    # Some districts were re-assigned between states/UTs (e.g. Leh and Kargil to Ladakh in 2019).
    overrides = aliases["hmis_state_overrides"]
    hmis["_state_norm"] = [
        overrides.get(district, state) for district, state in zip(hmis["_district_norm"], hmis["_state_norm"])
    ]
    hmis = aggregate_split_districts(hmis, aliases)

    hmis_by_state = {state: group.index.tolist() for state, group in hmis.groupby("_state_norm", dropna=False)}
    hmis_lgd: dict[str, int] = {}
    can_use_lgd = "lgd_code" in nfhs.columns and "lgd_code" in hmis.columns
    if can_use_lgd:
        hmis_lgd = {str(value): index for index, value in hmis["lgd_code"].items() if pd.notna(value)}

    used_hmis_rows: set[int] = set()
    matches: list[int | None] = []
    audit_rows: list[dict[str, object]] = []
    # Exact/LGD matches first so that fuzzy matching cannot steal an exact partner.
    order = sorted(
        nfhs.index,
        key=lambda index: 0
        if nfhs.at[index, "_district_norm"]
        in {hmis.at[i, "_district_norm"] for i in hmis_by_state.get(nfhs.at[index, "_state_norm"], [])}
        else 1,
    )
    decided: dict[int, tuple[int | None, str, float | None, str | None]] = {}
    for nfhs_index in order:
        row = nfhs.loc[nfhs_index]
        candidate: int | None = None
        method, score, best_name = "unmatched", None, None
        if can_use_lgd and pd.notna(row.get("lgd_code")):
            candidate = hmis_lgd.get(str(row["lgd_code"]))
            if candidate is not None and candidate not in used_hmis_rows:
                method, score = "lgd_code", 100.0
            else:
                candidate = None
        state_candidates = [i for i in hmis_by_state.get(row["_state_norm"], []) if i not in used_hmis_rows]
        if candidate is None and state_candidates:
            exact = [i for i in state_candidates if hmis.at[i, "_district_norm"] == row["_district_norm"]]
            if len(exact) == 1:
                candidate, method, score = exact[0], "exact_normalised_name", 100.0
            else:
                choices = {i: hmis.at[i, "_district_norm"] for i in state_candidates}
                result = process.extractOne(row["_district_norm"], choices, scorer=fuzz.token_sort_ratio)
                if result:
                    best_name, score, best_index = result
                    if score >= automatic_threshold:
                        candidate, method = best_index, "fuzzy_auto"
                    elif score >= review_threshold:
                        method = "manual_review"
        if candidate is not None:
            used_hmis_rows.add(candidate)
        decided[nfhs_index] = (candidate, method, score, best_name)

    for nfhs_index, row in nfhs.iterrows():
        candidate, method, score, best_name = decided[nfhs_index]
        matches.append(candidate)
        audit_rows.append(
            {
                "nfhs_state": row["state"],
                "nfhs_district": row["district"],
                "hmis_state": hmis.at[candidate, "state"] if candidate is not None else None,
                "hmis_district": hmis.at[candidate, "district"] if candidate is not None else None,
                "method": method,
                "similarity_score": score,
                "best_unaccepted_candidate": best_name if candidate is None else None,
            }
        )
    for index in sorted(set(hmis.index) - used_hmis_rows):
        audit_rows.append(
            {
                "nfhs_state": None,
                "nfhs_district": None,
                "hmis_state": hmis.at[index, "state"],
                "hmis_district": hmis.at[index, "district"],
                "method": "hmis_unused",
                "similarity_score": None,
                "best_unaccepted_candidate": None,
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)

    matched_share = sum(match is not None for match in matches) / max(len(matches), 1)
    if matched_share < minimum_matched_share:
        raise ValueError(
            f"Only {matched_share:.1%} of NFHS districts matched HMIS records (minimum "
            f"{minimum_matched_share:.0%}). Review {audit_path} and extend config/district_aliases.json."
        )

    payload_columns = [c for c in hmis.columns if c.startswith("hmis_")]
    joined = pd.DataFrame(
        [hmis.loc[m, payload_columns].to_dict() if m is not None else {} for m in matches],
        columns=payload_columns,
    )
    master = pd.concat([nfhs.drop(columns=["_district_norm", "_state_norm"]), joined], axis=1)
    master["hmis_matched"] = [m is not None for m in matches]
    master["hmis_match_method"] = [decided[i][1] for i in nfhs.index]
    if master.duplicated(["state", "district"]).any():
        raise AssertionError("Merge produced duplicate district keys.")
    return master
