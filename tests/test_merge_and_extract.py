import json

import pandas as pd

from src.data.hmis_extract import parse_state_report
from src.data.merge_nfhs_hmis import merge_sources


def _hmis_row(state, district, deaths=5, births=1000):
    return {
        "state": state,
        "district": district,
        "hmis_maternal_deaths": deaths,
        "hmis_live_births": births,
        "hmis_institutional_deliveries": births,
        "hmis_obstetric_complications": 50,
    }


def test_merge_aliases_aggregation_and_urban_rural(tmp_path):
    nfhs = pd.DataFrame(
        {
            "state": ["Karnataka", "Karnataka", "Haryana", "West Bengal", "Assam"],
            "district": ["Bangalore", "Bangalore Rural", "Gurgaon", "Jalpaiguri", "Majuli"],
            "institutional_births_pct": [95, 92, 90, 85, 70],
        }
    )
    hmis = pd.DataFrame(
        [
            _hmis_row("Karnataka", "Bangalore Urban", births=5000),
            _hmis_row("Karnataka", "Bangalore Rural", births=800),
            _hmis_row("Haryana", "Gurugram"),
            _hmis_row("West Bengal", "Jalpaiguri", births=1000),
            _hmis_row("West Bengal", "Alipurduar", births=400),
        ]
    )
    aliases = {
        "states": {},
        "districts": {"bangalore urban": "bangalore", "gurgaon": "gurugram"},
        "hmis_aggregate": {"west bengal": {"jalpaiguri": ["jalpaiguri", "alipurduar"]}},
    }
    (tmp_path / "aliases.json").write_text(json.dumps(aliases))
    nfhs.to_csv(tmp_path / "nfhs.csv", index=False)
    hmis.to_csv(tmp_path / "hmis.csv", index=False)
    master = merge_sources(
        tmp_path / "nfhs.csv", tmp_path / "hmis.csv", tmp_path / "aliases.json", tmp_path / "audit.csv", minimum_matched_share=0.5
    )
    births = dict(zip(master["district"], master["hmis_live_births"]))
    assert births["Bangalore"] == 5000 and births["Bangalore Rural"] == 800, "urban/rural pairs must not collapse"
    assert births["Jalpaiguri"] == 1400, "carved-out HMIS district must be summed into the NFHS parent"
    assert master.set_index("district").loc["Gurgaon", "hmis_matched"]
    assert not master.set_index("district").loc["Majuli", "hmis_matched"], "unmatched districts are kept but flagged"
    audit = pd.read_csv(tmp_path / "audit.csv")
    assert (audit["method"] == "unmatched").sum() == 1


def test_streaming_hmis_parser_handles_rowspan_districts_and_total_rows():
    header = '<td class="rowheader" rowspan="3">{}</td>'
    cell = '<td class="rowheader">{}</td>'
    data = "".join(f'<td class="data">{v}</td>\n' for v in [1, 2, 3, 4, 5, 999, 0, 0, 0, 0])
    lines = [
        "<table>\n",
        "<tr>\n", header.format("North") + "\n", cell.format("'2.2'") + "\n", cell.format("Live Birth - Male") + "\n", cell.format("TOTAL") + "\n", data, "</tr>\n",
        "<tr>\n", cell.format("'9'") + "\n", cell.format("Live Birth - Male") + "\n", cell.format("1. Stock") + "\n", data, "</tr>\n",
        "<tr>\n", cell.format("'2.3'") + "\n", cell.format("Live Birth - Female") + "\n", cell.format("TOTAL") + "\n", data.replace("999", "1,001"), "</tr>\n",
    ]
    rows = parse_state_report(iter(lines), "Demo", {"Live Birth - Male", "Live Birth - Female"})
    assert list(rows["district"]) == ["North", "North"]
    assert list(rows["annual_total"]) == [999.0, 1001.0], "annual total is the first of the last five data cells"
