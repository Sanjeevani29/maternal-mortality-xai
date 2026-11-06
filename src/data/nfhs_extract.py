from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# Canonical column -> NFHS-5 district factsheet indicator label (whitespace-normalised).
# The district factsheets do not publish JSY assistance, SC/ST share, rural share or male literacy,
# so those blueprint features are intentionally absent rather than approximated.
NFHS5_INDICATORS = {
    # Stage-1 anchor (forbidden as a Stage-2 feature)
    "institutional_births_pct": "Institutional births (%)",
    "institutional_births_public_pct": "Institutional births in public facility (%)",
    # D1 Emergency & facility quality
    "public_c_section_pct": "Births in a public health facility that were delivered by caesarean section (%)",
    "private_c_section_pct": "Births in a private health facility that were delivered by caesarean section (%)",
    "skilled_birth_attendance_pct": "Births attended by skilled health personnel (%)",
    # D2 Maternal health & nutrition
    "pregnant_anaemia_pct": "Pregnant women age 15-49 years who are anaemic",
    "all_women_anaemia_pct": "All women age 15-49 years who are anaemic (%)",
    "low_bmi_pct": "Women whose Body Mass Index (BMI) is below normal (BMI <18)",
    "teenage_pregnancy_pct": "Women age 15-19 years who were already mothers or pregnant at the time of the survey (%)",
    "married_before_18_pct": "Women age 2024 years married before age 18 years (%)",
    "ifa_100_days_pct": "Mothers who consumed iron folic acid for 100 days or more when they were pregnant (%)",
    "high_parity_pct": "Births in the 5 years preceding the survey that are third or higher order (%)",
    "female_elevated_bp_pct": "Female Elevated blood pressure or taking medicine to control blood pressure (%)",
    "female_high_blood_sugar_pct": "Female Blood sugar level high or very high (>140 mg/dl) or taking medicine to control blood sugar level (%)",
    # D3 Healthcare access & delivery
    "anc_4_plus_pct": "Mothers who had at least 4 antenatal care visits (%)",
    "first_trimester_anc_pct": "Mothers who had an antenatal checkup in the first trimester (%)",
    "postnatal_48h_pct": "Mothers who received postnatal care from a doctor/nurse/LHV/ANM/midwife/other health personnel within 2 days of delivery (%)",
    "oop_expenditure_inr": "Average out-of-pocket expenditure per delivery in a public health facility (Rs)",
    "tetanus_protected_pct": "Mothers whose last birth was protected against neonatal tetanus (%)",
    "mcp_card_pct": "Registered pregnancies for which the mother received a Mother and Child Protection (MCP) card (%)",
    # D4 Socioeconomic & education
    "female_literacy_pct": "Women who are literate (%)",
    "women_10y_schooling_pct": "Women with 10 or more years of schooling (%)",
    "female_ever_schooled_pct": "Female population age 6 years and above who ever attended school (%)",
    "health_insurance_pct": "Households with any usual member covered under a health insurance/financing scheme (%)",
    "death_registration_pct": "Deaths in the last 3 years registered with the civil authority (%)",
    # D5 Infrastructure & WASH
    "electricity_pct": "Population living in households with electricity (%)",
    "improved_water_pct": "Population living in households with an improved drinkingwater source (%)",
    "improved_sanitation_pct": "Population living in households that use an improved sanitation facility (%)",
    "clean_cooking_fuel_pct": "Households using clean fuel for cooking (%)",
}


def _normalise_indicator(value: object) -> str:
    text = str(value).replace("\\n", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _parse_value(value: object) -> float | None:
    """Parse factsheet values. Parenthesised values (25-49 unweighted cases) are kept; '*' is suppressed."""
    text = str(value).replace(",", "").strip()
    if text in {"", "nan", "NA", "N/A", "*", "na"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def build_nfhs_canonical(source_path: Path, output_path: Path) -> pd.DataFrame:
    """Convert the published long-form NFHS-5 factsheet extract into one record per district."""
    source = pd.read_csv(source_path, low_memory=False)
    required = {"State", "ST_CEN_CD", "District", "DT_CEN_CD", "Indicator", "NFHS 5"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"NFHS source is missing columns: {sorted(missing)}")
    source = source.copy()
    source["state"] = source["State"].astype(str).str.strip()
    source["district"] = source["District"].astype(str).str.strip()
    source["_indicator"] = source["Indicator"].map(_normalise_indicator)
    source["_value"] = source["NFHS 5"].map(_parse_value)
    inverse = {_normalise_indicator(label): column for column, label in NFHS5_INDICATORS.items()}
    selected = source.loc[source["_indicator"].isin(inverse)].copy()
    selected["_column"] = selected["_indicator"].map(inverse)
    wide = selected.pivot_table(
        index=["state", "district"], columns="_column", values="_value", aggfunc="first"
    ).reset_index()
    # The community mirror carries inconsistent census codes; keep them only as provenance.
    codes = source.groupby(["state", "district"], as_index=False).agg(
        census_state_code=("ST_CEN_CD", "first"), census_district_code=("DT_CEN_CD", "first")
    )
    wide = codes.merge(wide, on=["state", "district"], how="right")
    for column in ("census_state_code", "census_district_code"):
        wide[column] = pd.to_numeric(wide[column], errors="coerce").astype("Int64")
    absent = sorted(set(NFHS5_INDICATORS) - set(wide.columns))
    if absent:
        raise ValueError(f"NFHS indicator mapping no longer resolves: {absent}")
    if wide.duplicated(["state", "district"]).any():
        raise ValueError("NFHS canonical extract has duplicate state/district keys.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output_path, index=False)
    return wide
