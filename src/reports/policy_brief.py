"""Exportable policy briefs (Excel workbook and PDF) built from the master priority action table."""
from __future__ import annotations

import io
from datetime import date

import pandas as pd

from src.configuration import pretty

DISCLAIMER = (
    "Research decision-support output. Classes are derived from facility-reported HMIS outcomes and "
    "NFHS-5 survey indicators; SHAP drivers are statistical associations, not clinical causes. Validate "
    "with MDSR, CRS/verbal-autopsy and LaQshya facility audits before allocating resources."
)


def to_excel(actions: pd.DataFrame, domain_shares: pd.DataFrame | None = None) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        actions.to_excel(writer, sheet_name="Priority table", index=False)
        if domain_shares is not None and not domain_shares.empty:
            domain_shares.to_excel(writer, sheet_name="National domain shares", index=False)
        pd.DataFrame({"note": [DISCLAIMER]}).to_excel(writer, sheet_name="Read me", index=False)
    return buffer.getvalue()


def _latin1(text: object) -> str:
    return str(text).encode("latin-1", "replace").decode("latin-1")


def to_pdf(actions: pd.DataFrame, title: str, max_districts: int = 40) -> bytes:
    """A compact state policy brief: summary counts plus one block per highest-risk district."""
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(width, 7, _latin1(title))
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(width, 5, _latin1(f"Generated {date.today().isoformat()} | {len(actions)} districts in scope"))
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(width, 4, _latin1(DISCLAIMER))
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(width, 6, "Districts by operational class", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for label, count in actions["priority_label"].value_counts().items():
        pdf.cell(width, 5, _latin1(f"  {label}: {count}"), new_x="LMARGIN", new_y="NEXT")
    if "crs_verbal_autopsy_audit_flag" in actions:
        flagged = int(actions["crs_verbal_autopsy_audit_flag"].astype(bool).sum())
        pdf.cell(width, 5, _latin1(f"  Low-burden districts flagged for CRS / verbal-autopsy audit: {flagged}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    ranked = actions.sort_values("priority_risk_score", ascending=False).head(max_districts)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(width, 6, _latin1(f"Highest-risk districts (top {len(ranked)})"), new_x="LMARGIN", new_y="NEXT")
    for _, row in ranked.iterrows():
        pdf.set_font("Helvetica", "B", 9)
        pdf.multi_cell(
            width,
            5,
            _latin1(
                f"{row['district']}, {row['state']} - {row['priority_label']} | risk score {row['priority_risk_score']:.1f}% "
                f"| residual {row['studentized_residual']:+.2f}"
            ),
        )
        pdf.set_font("Helvetica", "", 8)
        domain = str(row.get("primary_domain_bottleneck", ""))
        pdf.multi_cell(width, 4, _latin1(f"Primary bottleneck: {pretty(domain)} -> {row.get('recommended_scheme', '')}"))
        for rank in range(1, 6):
            driver = row.get(f"top_{rank}_driver")
            if isinstance(driver, str) and driver:
                pdf.multi_cell(width, 4, _latin1(f"   {rank}. {driver}"))
        pdf.ln(1.5)
    return bytes(pdf.output())
