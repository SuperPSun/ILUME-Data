"""Plot coverage summaries for structured simulation datasets."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raw_prep import canonicalize_smiles, disable_rdkit_logs


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "structured" / "simulation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "simulation"
EXCLUDED_BAR_FILENAMES = {"simulated_combi_qm_solv_structured.csv"}
FINGERPRINT_BITS = 1024
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FINGERPRINT_BITS)


@dataclass(frozen=True)
class DatasetSummary:
    filename: str
    property_name: str
    dataset_label: str
    entry_count: int
    covered_entity_count: int
    entity_type: str
    included_in_scatter: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-max-iter", type=int, default=1000)
    return parser.parse_args()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


@lru_cache(maxsize=None)
def canonicalize_cached(smiles: str) -> str:
    return canonicalize_smiles(smiles)


def nonempty_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[column].map(clean_text)


def first_property_name(df: pd.DataFrame, fallback: str) -> str:
    if "property_name" not in df.columns:
        return dataset_label_from_filename(fallback)
    values = df["property_name"].map(clean_text)
    values = values[values != ""]
    if values.empty:
        return dataset_label_from_filename(fallback)
    return values.iloc[0]


def dataset_label_from_filename(filename_or_stem: str) -> str:
    stem = Path(filename_or_stem).stem
    stem = stem.removeprefix("simulated_").removesuffix("_structured")
    stem = stem.replace("_20260514_mapping", "")
    return stem


def canonical_series(values: pd.Series) -> pd.Series:
    return values.map(canonicalize_cached)


def unique_nonempty_pairs(left: pd.Series, right: pd.Series) -> int:
    mask = (left != "") & (right != "")
    if not mask.any():
        return 0
    return (left[mask] + "||" + right[mask]).nunique()


def classify_entities(path: Path, df: pd.DataFrame) -> tuple[int, str, bool]:
    cation = nonempty_series(df, "cation")
    anion = nonempty_series(df, "anion")
    pair_count = unique_nonempty_pairs(cation, anion)
    if pair_count:
        return pair_count, "cation+anion", True

    cation_count = cation[cation != ""].nunique()
    if cation_count:
        return cation_count, "cation", False

    anion_count = anion[anion != ""].nunique()
    if anion_count:
        return anion_count, "anion", False

    solvent = nonempty_series(df, "solvent")
    solute = nonempty_series(df, "solute")
    solvation_count = unique_nonempty_pairs(solvent, solute)
    if solvation_count:
        return solvation_count, "solvent+solute", False

    smiles = nonempty_series(df, "SMILES")
    smiles_count = smiles[smiles != ""].nunique()
    if smiles_count:
        return smiles_count, "SMILES", False

    return 0, "unclassified", False


def load_datasets(input_dir: Path) -> list[tuple[Path, pd.DataFrame]]:
    paths = sorted(input_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    return [(path, pd.read_csv(path)) for path in paths]


def summarize_datasets(datasets: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    summaries: list[DatasetSummary] = []
    for path, df in datasets:
        property_name = first_property_name(df, path.name)
        count, entity_type, scatterable = classify_entities(path, df)
        summaries.append(
            DatasetSummary(
                filename=path.name,
                property_name=property_name,
                dataset_label=property_name,
                entry_count=len(df),
                covered_entity_count=count,
                entity_type=entity_type,
                included_in_scatter=scatterable,
            )
        )
    return pd.DataFrame(summaries)


def collect_paired_records(datasets: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    paired_rows: list[pd.DataFrame] = []

    for path, df in datasets:
        property_name = first_property_name(df, path.name)
        cation = canonical_series(nonempty_series(df, "cation"))
        anion = canonical_series(nonempty_series(df, "anion"))
        pair_mask = (cation != "") & (anion != "")

        if pair_mask.any():
            pair_df = pd.DataFrame(
                {
                    "property_name": property_name,
                    "cation": cation[pair_mask],
                    "anion": anion[pair_mask],
                    "point_type": "cation+anion",
                }
            ).drop_duplicates()
            paired_rows.append(pair_df)

    if not paired_rows:
        return pd.DataFrame()
    return pd.concat(paired_rows, ignore_index=True).drop_duplicates()


@lru_cache(maxsize=None)
def smiles_fingerprint(smiles: str) -> tuple[int, ...]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return tuple([0] * FINGERPRINT_BITS)
    fingerprint = MORGAN_GENERATOR.GetFingerprint(mol)
    array = np.zeros((FINGERPRINT_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return tuple(int(value) for value in array)


def build_tsne_scatter(records: pd.DataFrame, *, perplexity: float, max_iter: int) -> pd.DataFrame:
    unique_pairs = records[["cation", "anion"]].drop_duplicates().reset_index(drop=True)
    features = np.zeros((len(unique_pairs), FINGERPRINT_BITS * 2), dtype=np.float32)

    for index, row in unique_pairs.iterrows():
        features[index, :FINGERPRINT_BITS] = smiles_fingerprint(row["cation"])
        features[index, FINGERPRINT_BITS:] = smiles_fingerprint(row["anion"])

    pca_components = min(50, len(unique_pairs), features.shape[1])
    reduced = PCA(n_components=pca_components, random_state=0).fit_transform(features)
    effective_perplexity = min(perplexity, max(1.0, (len(unique_pairs) - 1) / 3))
    coords = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=max_iter,
        random_state=0,
        n_jobs=-1,
    ).fit_transform(reduced)
    unique_pairs["TSNE1"] = coords[:, 0]
    unique_pairs["TSNE2"] = coords[:, 1]
    return records.merge(unique_pairs, on=["cation", "anion"], how="left")


def plot_bar(summary: pd.DataFrame, output_path: Path, dpi: int) -> None:
    plot_df = summary[~summary["filename"].isin(EXCLUDED_BAR_FILENAMES)]
    plot_df = plot_df.sort_values("entry_count", ascending=False).reset_index(drop=True)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(15, 7.8))
    x_pos = np.arange(len(plot_df))
    width = 0.38
    colors = ["#3B6EA8", "#E18A2D"]

    bars_entries = ax.bar(
        x_pos - width / 2,
        plot_df["entry_count"],
        width,
        color=colors[0],
        alpha=0.9,
        label="Entry count",
    )
    bars_entities = ax.bar(
        x_pos + width / 2,
        plot_df["covered_entity_count"],
        width,
        color=colors[1],
        alpha=0.9,
        label="IL/Cation/Anion count",
    )

    ax.bar_label(bars_entries, labels=[f"{value / 1000:.1f}k" for value in plot_df["entry_count"]], fontsize=8, padding=3)
    ax.bar_label(
        bars_entities,
        labels=[f"{value / 1000:.1f}k" for value in plot_df["covered_entity_count"]],
        fontsize=8,
        padding=3,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_df["dataset_label"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.set_title("Simulation Dataset Coverage", pad=18, weight="bold")
    ax.legend(frameon=False, ncols=2, loc="upper right")
    ax.grid(axis="y", color="#D7DBDF", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    sns.despine(fig=fig, ax=ax, left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(scatter: pd.DataFrame, output_path: Path, dpi: int) -> None:
    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(12.5, 9))
    properties = sorted(scatter["property_name"].unique())
    palette = dict(zip(properties, sns.color_palette("Set2", n_colors=len(properties))))

    for property_name, group in scatter.groupby("property_name", sort=True):
        ax.scatter(
            group["TSNE1"],
            group["TSNE2"],
            s=9,
            alpha=0.46,
            linewidths=0.15,
            edgecolors="white",
            color=palette[property_name],
            label=property_name,
            rasterized=True,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Simulation Property Distribution in Ion-Pair t-SNE Space", pad=16, weight="bold")
    ax.grid(color="#E2E6EA", linewidth=0.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=2.6)
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    disable_rdkit_logs()

    datasets = load_datasets(input_dir)
    summary = summarize_datasets(datasets)
    paired_records = collect_paired_records(datasets)

    summary_path = output_dir / "simulation_coverage_summary.csv"
    bar_path = output_dir / "simulation_coverage_bar.png"
    scatter_path = output_dir / "simulation_property_distribution_scatter.png"

    summary.to_csv(summary_path, index=False)
    plot_bar(summary, bar_path, args.dpi)
    if paired_records.empty:
        raise ValueError("No cation/anion paired data found for scatter plot.")
    scatter = build_tsne_scatter(
        paired_records,
        perplexity=args.tsne_perplexity,
        max_iter=args.tsne_max_iter,
    )
    plot_scatter(scatter, scatter_path, args.dpi)

    print(f"Wrote {summary_path}")
    print(f"Wrote {bar_path}")
    print(f"Wrote {scatter_path}")
    print(f"Scatter points: {len(scatter)}")
    print(f"Unique ion pairs in t-SNE: {scatter[['cation', 'anion']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
