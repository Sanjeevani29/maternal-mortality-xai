from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEMO_STATES = [
    "Demo North",
    "Demo South",
    "Demo East",
    "Demo West",
    "Demo Central",
    "Demo Hills",
    "Demo Coast",
    "Demo Plains",
]


def write_demo_sources(nfhs_path: Path, hmis_path: Path, n_rows: int, random_state: int) -> None:
    """Create clearly SYNTHETIC inputs in the same canonical format as the real-data extractors.

    They exist only to exercise the software end to end and must never be read as district evidence.
    """
    rng = np.random.default_rng(random_state)
    nfhs_rows: list[dict[str, object]] = []
    hmis_rows: list[dict[str, object]] = []

    def pct(centre: float, spread: float, low: float = 0.5, high: float = 99.5) -> float:
        return float(np.clip(centre + rng.normal(0, spread), low, high))

    for index in range(n_rows):
        state = DEMO_STATES[index % len(DEMO_STATES)]
        quality = rng.normal(0, 1)  # latent facility-quality / nutrition deficit (higher = worse)
        wealth = rng.normal(0, 1)
        care = float(np.clip(78 + 9 * wealth + rng.normal(0, 7), 35, 100))
        births = int(rng.integers(8_000, 60_000))
        mmr = float(np.clip(260 - 1.6 * care + 55 * quality + rng.normal(0, 15), 20, 550))
        morbidity = float(np.clip(6 + 0.04 * (100 - care) + 2.5 * quality + rng.normal(0, 1), 0.3, 25))
        district = f"Demo District {index + 1:03d}"
        nfhs_rows.append(
            {
                "state": state,
                "district": district,
                "institutional_births_pct": round(care, 2),
                "institutional_births_public_pct": round(care * 0.7, 2),
                "public_c_section_pct": pct(15 - 4 * quality, 4),
                "private_c_section_pct": pct(42 - 3 * quality, 9),
                "skilled_birth_attendance_pct": pct(care - 2, 5),
                "pregnant_anaemia_pct": pct(47 + 7 * quality, 7),
                "all_women_anaemia_pct": pct(55 + 5 * quality, 6),
                "low_bmi_pct": pct(19 + 3 * quality - 2 * wealth, 3),
                "teenage_pregnancy_pct": pct(6 + 1.5 * quality - wealth, 2),
                "married_before_18_pct": pct(20 + 4 * quality - 4 * wealth, 5),
                "ifa_100_days_pct": pct(48 - 7 * quality, 8),
                "female_elevated_bp_pct": pct(21 + 2 * quality, 4),
                "female_high_blood_sugar_pct": pct(12 + 2 * wealth, 3),
                "anc_4_plus_pct": pct(care - 18 - 3 * quality, 8),
                "first_trimester_anc_pct": pct(70 - 5 * quality, 8),
                "postnatal_48h_pct": pct(care - 8 - 2 * quality, 6),
                "oop_expenditure_inr": float(np.clip(3000 + 600 * quality + rng.normal(0, 800), 300, 15_000)),
                "tetanus_protected_pct": pct(91 - 2 * quality, 3),
                "mcp_card_pct": pct(95 - 2 * quality, 3),
                "female_literacy_pct": pct(72 + 8 * wealth, 6),
                "women_10y_schooling_pct": pct(40 + 10 * wealth, 7),
                "female_ever_schooled_pct": pct(71 + 8 * wealth, 6),
                "health_insurance_pct": pct(38 + 6 * wealth, 10),
                "death_registration_pct": pct(75 + 8 * wealth, 9),
                "electricity_pct": pct(96 + 2 * wealth, 2),
                "improved_water_pct": pct(94 + 2 * wealth, 3),
                "improved_sanitation_pct": pct(70 + 8 * wealth - 2 * quality, 7),
                "clean_cooking_fuel_pct": pct(50 + 12 * wealth, 9),
            }
        )
        deliveries = max(1, round(births * care / 100))
        anc = round(births * 1.15)
        record = {
            "state": state,
            "district": district,
            "hmis_live_births": births,
            "hmis_maternal_deaths": int(rng.poisson(mmr * births / 100_000)),
            "hmis_institutional_deliveries": deliveries,
            "hmis_obstetric_complications": round(deliveries * morbidity / 100),
            "hmis_total_c_sections": round(deliveries * pct(18 - 3 * quality, 5) / 100),
            "hmis_anc_registered": anc,
            "hmis_anc_first_trimester": round(anc * pct(72 - 5 * quality, 8) / 100),
            "hmis_ifa_180": round(anc * pct(85 - 6 * quality, 8) / 100),
            "hmis_hb_tested_4plus": round(anc * pct(80, 8) / 100),
            "hmis_hb_below_7": round(anc * pct(3 + 1.5 * quality, 1.0, 0.1, 30) / 100),
            "hmis_blood_units_issued": round(deliveries * max(5, 160 - 45 * quality + rng.normal(0, 30)) / 1000),
            "hmis_discharged_within_48h": round(deliveries * pct(35 + 8 * quality, 8) / 100),
            "hmis_jssk_transport": round(deliveries * pct(30 - 6 * quality, 8) / 100),
            "hmis_years_pooled": 3,
        }
        # A little missingness so the imputation path is always exercised.
        if index % 17 == 0:
            nfhs_rows[-1]["pregnant_anaemia_pct"] = np.nan
        if index % 23 == 0:
            record["hmis_blood_units_issued"] = np.nan
        hmis_rows.append(record)
    nfhs_path.parent.mkdir(parents=True, exist_ok=True)
    hmis_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(nfhs_rows).to_csv(nfhs_path, index=False)
    pd.DataFrame(hmis_rows).to_csv(hmis_path, index=False)
