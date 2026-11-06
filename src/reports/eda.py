"""Publication figures for Step 4 (EDA): quadrant plot, state distribution, domain heatmap, violins."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.configuration import domain_by_feature, feature_domains, pretty
from src.target.ols_priority_classifier import CLASS_NAMES

CLASS_COLOURS = {1: "#D32F2F", 2: "#F57C00", 3: "#388E3C", 4: "#1976D2"}


def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    return plt


def domain_scores(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Mean z-score of each domain's member features (descriptive only; orientation is not adjusted)."""
    lookup = domain_by_feature()
    scores = {}
    for domain in feature_domains():
        members = [feature for feature in feature_columns if lookup.get(feature) == domain]
        if not members:
            continue
        block = frame[members].apply(pd.to_numeric, errors="coerce")
        z = (block - block.mean()) / block.std(ddof=0).replace(0, np.nan)
        scores[domain] = z.mean(axis=1)
    return pd.DataFrame(scores, index=frame.index)


def quadrant_plot(frame: pd.DataFrame, stage1: dict, path: Path) -> Path:
    plt = _setup()
    figure, axis = plt.subplots(figsize=(7.5, 5))
    for label, group in frame.groupby("priority_class"):
        axis.scatter(
            group["institutional_births_pct"],
            group["composite_maternal_risk"],
            s=14,
            alpha=0.75,
            color=CLASS_COLOURS[int(label)],
            label=CLASS_NAMES[int(label)],
            edgecolor="white",
            linewidth=0.3,
        )
    x = np.linspace(frame["institutional_births_pct"].min(), frame["institutional_births_pct"].max(), 100)
    axis.plot(x, stage1["intercept"] + stage1["coefficient"] * x, color="#333", linewidth=1.2, label="OLS expected risk")
    axis.axvline(stage1["care_threshold"], color="#888", linestyle="--", linewidth=0.8)
    axis.axhline(stage1["risk_median"], color="#888", linestyle=":", linewidth=0.8)
    axis.set_xlabel("Institutional births (%)")
    axis.set_ylabel("Composite maternal risk (z)")
    axis.set_title("OLS discordance quadrants")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def state_distribution(frame: pd.DataFrame, path: Path) -> Path:
    plt = _setup()
    counts = frame.groupby("state")["priority_class"].apply(lambda s: (s == 1).sum()).sort_values()
    counts = counts[counts > 0]
    figure, axis = plt.subplots(figsize=(7, max(3, 0.22 * len(counts))))
    axis.barh(counts.index, counts.to_numpy(), color=CLASS_COLOURS[1])
    axis.set_xlabel("Number of Priority / High-Risk districts")
    axis.set_title("State distribution of Priority / High-Risk (class 1) districts")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def domain_heatmap(frame: pd.DataFrame, feature_columns: list[str], path: Path) -> Path:
    plt = _setup()
    import seaborn as sns

    lookup = domain_by_feature()
    ordered = [feature for domain in feature_domains() for feature in feature_columns if lookup.get(feature) == domain]
    correlation = frame[ordered].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    size = max(6, 0.3 * len(ordered))
    figure, axis = plt.subplots(figsize=(size + 1.5, size))
    sns.heatmap(correlation, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, ax=axis, cbar_kws={"shrink": 0.6})
    labels = [pretty(feature) for feature in ordered]
    axis.set_xticks(np.arange(len(ordered)) + 0.5, labels, rotation=90, fontsize=7)
    axis.set_yticks(np.arange(len(ordered)) + 0.5, labels, rotation=0, fontsize=7)
    axis.set_title("Spearman correlation of features (grouped by domain)")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def domain_violins(frame: pd.DataFrame, feature_columns: list[str], path: Path) -> Path:
    plt = _setup()
    import seaborn as sns

    scores = domain_scores(frame, feature_columns)
    scores["class"] = frame["priority_class"].map(CLASS_NAMES)
    subset = scores.loc[frame["priority_class"].isin([1, 3])]
    long = subset.melt(id_vars="class", var_name="domain", value_name="score")
    long["domain"] = long["domain"].map(lambda value: value.replace("_", " ").title())
    figure, axis = plt.subplots(figsize=(9, 4.2))
    sns.violinplot(
        data=long,
        x="domain",
        y="score",
        hue="class",
        split=True,
        inner="quart",
        palette={CLASS_NAMES[1]: CLASS_COLOURS[1], CLASS_NAMES[3]: CLASS_COLOURS[3]},
        ax=axis,
        cut=0,
    )
    axis.set_xlabel("")
    axis.set_ylabel("Mean member-feature z-score")
    axis.set_title("Priority / High-Risk vs Best Performer districts across the five domains")
    axis.tick_params(axis="x", labelsize=8)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def wash_vs_residual(frame: pd.DataFrame, path: Path) -> Path | None:
    if "wash_deprivation_score" not in frame:
        return None
    plt = _setup()
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(frame["wash_deprivation_score"], frame["studentized_residual"], s=10, alpha=0.6, color="#2a6f97")
    rho = frame[["wash_deprivation_score", "studentized_residual"]].corr(method="spearman").iloc[0, 1]
    axis.set_xlabel("WASH deprivation score")
    axis.set_ylabel("Studentized OLS residual")
    axis.set_title(f"WASH deprivation vs discordance (Spearman rho = {rho:.2f})")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def generate_eda_figures(frame: pd.DataFrame, feature_columns: list[str], stage1: dict, figure_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    written = [
        quadrant_plot(frame, stage1, figure_dir / "eda_ols_quadrants.png"),
        state_distribution(frame, figure_dir / "eda_state_class1_distribution.png"),
        domain_heatmap(frame, feature_columns, figure_dir / "eda_domain_correlation_heatmap.png"),
        domain_violins(frame, feature_columns, figure_dir / "eda_domain_violins_class1_vs_class3.png"),
        wash_vs_residual(frame, figure_dir / "eda_wash_vs_residual.png"),
    ]
    return [path for path in written if path is not None]
