"""Extract district annual totals from official HMIS "C2 item-wise monthly" archives.

The official ``.xls`` files are SAS-generated HTML tables of up to ~130 MB per state, so they are
parsed with a streaming line reader instead of ``pandas.read_html`` (which would materialise the
whole DOM). Each table row carries ``[district (rowspan)], item code, item name, type`` header cells
followed by 65 data cells: 13 periods (12 months + annual Total) x 5 sectors. The first of the final
five data cells is the all-sector annual total.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Iterable, Iterator
from zipfile import ZipFile

import pandas as pd


MATERNAL_DEATH_ITEMS = (
    "Number of Maternal Deaths due to Bleeding",
    "Number of Maternal Deaths due to High fever",
    "Number of Maternal Deaths due to Abortion",
    "Number of Maternal Deaths due to Obstructed/prolonged labour",
    "Number of Maternal Deaths due to Severe hypertesnion/fits",
    "Number of Maternal Deaths due to Other Causes (including causes not known)",
)

# Items required for Stage-1 target construction. Extraction fails loudly if any is absent.
CORE_ITEMS = {
    "live_birth_male": "Live Birth - Male",
    "live_birth_female": "Live Birth - Female",
    "institutional_deliveries": "Number of Institutional Deliveries conducted (Including C-Sections)",
    "obstetric_complications": "Number of cases of pregnant women with Obstetric Complications attended (Antepartum haemorrhage (APH), Post-Partum Hemorrhage (PPH), Sepsis, Eclampsia and others)",
}

# Items used only to engineer Stage-2 explanatory features. Missing items become NaN.
FEATURE_ITEMS = {
    "total_c_sections": "Total C -Section deliveries performed",
    "anc_registered": "Total number of pregnant women registered for ANC",
    "anc_first_trimester": "Out of the total ANC registered, number registered within 1st trimester (within 12 weeks)",
    "ifa_180": "Number of PW given 180 Iron Folic Acid (IFA) tablets",
    "hb_tested_4plus": "Number of PW tested for Haemoglobin (Hb ) 4 or more than 4 times for respective ANCs",
    "hb_below_7": "Number of PW having Hb level<7 (tested cases)",
    "blood_units_issued": "Number of blood units issued",
    "discharged_within_48h": "Out of total institutional deliveries number of women discharged within 48 hours of delivery",
    "jssk_transport": "Number of Pregnant Women provided - Free Home to facility transport under JSSK",
}

_CELL = re.compile(r"<t[dh]\b([^>]*)>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_ROWSPAN = re.compile(r"rowspan\s*=\s*\"?(\d+)", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub("", text))).strip()


def _number(text: str) -> float | None:
    value = text.replace(",", "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _iter_rows(lines: Iterable[str]) -> Iterator[list[tuple[str, str, int]]]:
    """Yield table rows as lists of ``(css_class, text, rowspan)`` cells from a line stream."""
    row: list[tuple[str, str, int]] | None = None
    buffer = ""
    for line in lines:
        lowered = line.lower()
        if "<tr" in lowered:
            row, buffer = [], ""
        if row is None:
            continue
        buffer += line
        lowered_buffer = buffer.lower()
        if "</td>" in lowered_buffer or "</th>" in lowered_buffer:
            # Cells normally sit on one physical line; the buffer also tolerates wrapped cells.
            for attributes, content in _CELL.findall(buffer):
                css = re.search(r"class=\"([^\"]+)\"", attributes)
                span = _ROWSPAN.search(attributes)
                row.append((css.group(1) if css else "", _clean(content), int(span.group(1)) if span else 1))
            buffer = ""
        if "</tr>" in lowered:
            yield row
            row, buffer = None, ""


def parse_state_report(lines: Iterable[str], state: str, wanted_items: set[str]) -> pd.DataFrame:
    """Return long-form ``state, district, item, annual_total`` rows for the wanted items."""
    records: list[dict[str, object]] = []
    district: str | None = None
    for row in _iter_rows(lines):
        headers = [cell for cell in row if cell[0] == "rowheader"]
        data = [cell for cell in row if cell[0] == "data"]
        if len(headers) == 4:
            district = headers[0][1]
            item, kind = headers[2][1], headers[3][1]
        elif len(headers) == 3:
            item, kind = headers[1][1], headers[2][1]
        else:
            continue
        if district is None or kind.upper() != "TOTAL" or item not in wanted_items or len(data) < 5:
            continue
        records.append(
            {"state": state, "district": district, "item": item, "annual_total": _number(data[-5][1])}
        )
    return pd.DataFrame.from_records(records, columns=["state", "district", "item", "annual_total"])


# Map HMIS archive file names onto the NFHS-5 state spellings (NFHS-5 keeps DNH and Daman & Diu separate).
STATE_ALIASES = {
    "A & N Islands": "Andaman & Nicobar Island",
    "Delhi": "NCT of Delhi",
}


def _state_from_member(member: str) -> str:
    name = Path(member).stem.strip()
    return STATE_ALIASES.get(name, name)


def _state_members(archive: ZipFile) -> list[str]:
    return [
        member
        for member in archive.namelist()
        if member.lower().endswith(".xls") and "all_india" not in member.lower() and "/M O " not in member
    ]


def extract_hmis_annual(zip_path: Path, progress: bool = False) -> pd.DataFrame:
    """Extract one financial year of district totals without unpacking the multi-GB archive."""
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    wanted = set(CORE_ITEMS.values()) | set(FEATURE_ITEMS.values()) | set(MATERNAL_DEATH_ITEMS)
    frames: list[pd.DataFrame] = []
    with ZipFile(zip_path) as archive:
        for member in _state_members(archive):
            with archive.open(member) as handle:
                lines = (raw.decode("cp1252", errors="replace") for raw in handle)
                state_rows = parse_state_report(lines, _state_from_member(member), wanted)
            if progress:
                print(f"  {zip_path.name}: {member} -> {state_rows['district'].nunique()} districts", flush=True)
            if not state_rows.empty:
                frames.append(state_rows)
    if not frames:
        raise ValueError(f"No state reports found in {zip_path}")
    long = pd.concat(frames, ignore_index=True)
    pivot = long.pivot_table(
        index=["state", "district"], columns="item", values="annual_total", aggfunc="sum"
    ).reset_index()
    required = set(CORE_ITEMS.values()) | set(MATERNAL_DEATH_ITEMS)
    absent = required - set(pivot.columns)
    if absent:
        raise ValueError(f"HMIS archive {zip_path} is missing required indicators: {sorted(absent)}")

    output = pivot[["state", "district"]].copy()
    output["hmis_live_births"] = pivot[CORE_ITEMS["live_birth_male"]] + pivot[CORE_ITEMS["live_birth_female"]]
    output["hmis_maternal_deaths"] = pivot[list(MATERNAL_DEATH_ITEMS)].sum(axis=1, min_count=1)
    output["hmis_institutional_deliveries"] = pivot[CORE_ITEMS["institutional_deliveries"]]
    output["hmis_obstetric_complications"] = pivot[CORE_ITEMS["obstetric_complications"]]
    for name, label in FEATURE_ITEMS.items():
        output[f"hmis_{name}"] = pivot[label] if label in pivot else float("nan")
    # Drop state-total or blank rows that are not districts.
    output = output.loc[output["district"].str.len() > 0]
    return output


def pool_hmis_annual_files(
    yearly_files: list[Path], output_path: Path, progress: bool = False, aliases_path: Path | None = None
) -> pd.DataFrame:
    """Pool financial-year extracts as summed counts (never averaged rates).

    Only the archives that exist on disk are pooled; ``hmis_years_pooled`` records how many years each
    district contributed. Years are aligned on normalised state/district names (plus the curated
    aliases), so spelling changes between years do not split a district; the most recent spelling is kept.
    """
    present = [path for path in yearly_files if path.exists()]
    if not present:
        raise FileNotFoundError(f"None of the HMIS archives exist: {[str(path) for path in yearly_files]}")
    annual = []
    for path in present:
        current = extract_hmis_annual(path, progress=progress)
        current["source_year"] = path.stem
        annual.append(current)
    combined = pd.concat(annual, ignore_index=True)
    count_columns = [column for column in combined if column.startswith("hmis_")]
    if aliases_path is not None and aliases_path.exists():
        from src.data.merge_nfhs_hmis import load_aliases, normalise_district, normalise_state

        aliases = load_aliases(aliases_path)
        combined["_state_key"] = combined["state"].map(lambda value: normalise_state(value, aliases))
        combined["_district_key"] = combined["district"].map(lambda value: normalise_district(value, aliases))
    else:
        combined["_state_key"] = combined["state"].str.strip().str.lower()
        combined["_district_key"] = combined["district"].str.strip().str.lower()
    keys = ["_state_key", "_district_key"]
    pooled = combined.groupby(keys, as_index=False)[count_columns].sum(min_count=1)
    latest = combined.sort_values("source_year").groupby(keys, as_index=False)[["state", "district"]].last()
    years = combined.groupby(keys, as_index=False)["source_year"].nunique().rename(columns={"source_year": "hmis_years_pooled"})
    pooled = latest.merge(pooled, on=keys).merge(years, on=keys).drop(columns=keys)
    if pooled.duplicated(["state", "district"]).any():
        raise ValueError("Pooled HMIS table has duplicate state/district display names.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output_path, index=False)
    output_path.with_suffix(".years.json").write_text(
        json.dumps({"pooled_files": [str(path.name) for path in present]}, indent=2), encoding="utf-8"
    )
    return pooled
