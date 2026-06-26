"""Plot coverage summaries for structured AIonopedia and ILThermo datasets."""

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


DEFAULT_AIONOPEDIA_DIR = PROJECT_ROOT / "data" / "structured" / "AIonopedia"
DEFAULT_ILTHERMO_DIR = PROJECT_ROOT / "data" / "structured" / "ILThermo"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "aionopedia_ilthermo"
FINGERPRINT_BITS = 1024
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FINGERPRINT_BITS)

PROPERTY_LABELS_BY_FILENAME = {
    "AIonopedia_density_constructed.csv": "density",
    "AIonopedia_melt_constructed.csv": "melt",
    "AIonopedia_tension_constructed.csv": "tension",
    "AIonopedia_viscosity_constructed.csv": "viscosity",
    "ilt_density_structured.csv": "density",
    "ilt_normal_melting_temperature_structured.csv": "melt",
    "ilt_surface_tension_liquid_gas_structured.csv": "tension",
    "ilt_viscosity_structured.csv": "viscosity",
}


@dataclass(frozen=True)
class DatasetRecord:
    source: str
    filename: str
    property_name: str
    entry_count: int
    ion_pair_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aionopedia-dir", type=Path, default=DEFAULT_AIONOPEDIA_DIR)
    parser.add_argument("--ilthermo-dir", type=Path, default=DEFAULT_ILTHERMO_DIR)
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


def source_from_path(path: Path) -> str:
    parent = path.parent.name
    if parent in {"AIonopedia", "ILThermo"}:
        return parent
    return parent or "unknown"


def property_label_from_filename(filename: str) -> str:
    if filename in PROPERTY_LABELS_BY_FILENAME:
        return PROPERTY_LABELS_BY_FILENAME[filename]

    stem = Path(filename).stem
    if stem.startswith("AIonopedia_"):
        return stem.removeprefix("AIonopedia_").removesuffix("_constructed")
    if stem.startswith("ilt_"):
        return stem.removeprefix("ilt_").removesuffix("_structured")
    return stem


def canonical_pair_frame(df: pd.DataFrame) -> pd.DataFrame:
    cation = nonempty_series(df, "cation").map(canonicalize_cached)
    anion = nonempty_series(df, "anion").map(canonicalize_cached)
    mask = (cation != "") & (anion != "")
    return pd.DataFrame({"cation": cation[mask], "anion": anion[mask]}).drop_duplicates()


def load_datasets(input_dirs: list[Path]) -> list[tuple[Path, pd.DataFrame]]:
    datasets: list[tuple[Path, pd.DataFrame]] = []
    for input_dir in input_dirs:
        paths = sorted(input_dir.glob("*.csv"))
        if not paths:
            raise FileNotFoundError(f"No CSV files found in {input_dir}")
        for path in paths:
            df = pd.read_csv(path)
            if {"cation", "anion"}.issubset(df.columns):
                datasets.append((path, df))
    if not datasets:
        searched = ", ".join(str(path) for path in input_dirs)
        raise ValueError(f"No CSV files with cation/anion columns found in: {searched}")
    return datasets


def summarize_datasets(datasets: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    records: list[DatasetRecord] = []
    pairs_by_property: dict[str, list[pd.DataFrame]] = {}

    for path, df in datasets:
        property_name = property_label_from_filename(path.name)
        pairs = canonical_pair_frame(df)
        pairs_by_property.setdefault(property_name, []).append(pairs)
        records.append(
            DatasetRecord(
                source=source_from_path(path),
                filename=path.name,
                property_name=property_name,
                entry_count=len(df),
                ion_pair_count=len(pairs),
            )
        )

    file_summary = pd.DataFrame(records)
    property_summary = (
        file_summary.groupby("property_name", as_index=False)
        .agg(
            entry_count=("entry_count", "sum"),
            source_files=("filename", lambda values: "; ".join(sorted(values))),
            sources=("source", lambda values: "; ".join(sorted(set(values)))),
        )
        .sort_values("entry_count", ascending=False)
        .reset_index(drop=True)
    )
    property_summary["ion_pair_count"] = property_summary["property_name"].map(
        {
            property_name: pd.concat(pair_frames, ignore_index=True).drop_duplicates().shape[0]
            for property_name, pair_frames in pairs_by_property.items()
        }
    )
    return property_summary[["property_name", "entry_count", "ion_pair_count", "sources", "source_files"]]


def collect_paired_records(datasets: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    paired_rows: list[pd.DataFrame] = []
    for path, df in datasets:
        property_name = property_label_from_filename(path.name)
        pairs = canonical_pair_frame(df)
        if pairs.empty:
            continue
        pairs["property_name"] = property_name
        paired_rows.append(pairs)

    if not paired_rows:
        return pd.DataFrame(columns=["property_name", "cation", "anion"])
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
    if len(unique_pairs) < 2:
        raise ValueError("At least two unique ion pairs are required for t-SNE.")

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
    plot_df = summary.sort_values("entry_count", ascending=False).reset_index(drop=True)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(17, 8.5))
    x_pos = np.arange(len(plot_df))
    width = 0.38
    colors = ["#34699A", "#D9822B"]

    bars_entries = ax.bar(
        x_pos - width / 2,
        plot_df["entry_count"],
        width,
        color=colors[0],
        alpha=0.92,
        label="Entry count",
    )
    bars_pairs = ax.bar(
        x_pos + width / 2,
        plot_df["ion_pair_count"],
        width,
        color=colors[1],
        alpha=0.92,
        label="Unique ion pairs",
    )

    ax.bar_label(bars_entries, labels=[format_count(value) for value in plot_df["entry_count"]], fontsize=7, padding=3)
    ax.bar_label(bars_pairs, labels=[format_count(value) for value in plot_df["ion_pair_count"]], fontsize=7, padding=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_df["property_name"], rotation=38, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.set_title("AIonopedia + ILThermo Ion-Pair Coverage", pad=18, weight="bold")
    ax.legend(frameon=False, ncols=2, loc="upper right")
    ax.grid(axis="y", color="#D7DBDF", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    sns.despine(fig=fig, ax=ax, left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def format_count(value: int) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return str(value)


def plot_scatter(scatter: pd.DataFrame, output_path: Path, dpi: int) -> None:
    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(13, 9.5))
    properties = sorted(scatter["property_name"].unique())
    palette = dict(zip(properties, sns.color_palette("tab20", n_colors=len(properties))))

    for property_name, group in scatter.groupby("property_name", sort=True):
        ax.scatter(
            group["TSNE1"],
            group["TSNE2"],
            s=10,
            alpha=0.5,
            linewidths=0.15,
            edgecolors="white",
            color=palette[property_name],
            label=property_name,
            rasterized=True,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("AIonopedia + ILThermo Property Distribution in Ion-Pair t-SNE Space", pad=16, weight="bold")
    ax.grid(color="#E2E6EA", linewidth=0.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=2.5, fontsize=9)
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dirs = [args.aionopedia_dir.resolve(), args.ilthermo_dir.resolve()]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    disable_rdkit_logs()

    datasets = load_datasets(input_dirs)
    summary = summarize_datasets(datasets)
    paired_records = collect_paired_records(datasets)

    summary_path = output_dir / "aionopedia_ilthermo_coverage_summary.csv"
    bar_path = output_dir / "aionopedia_ilthermo_coverage_bar.png"
    scatter_path = output_dir / "aionopedia_ilthermo_property_tsne.png"

    summary.to_csv(summary_path, index=False)
    plot_bar(summary, bar_path, args.dpi)
    if paired_records.empty:
        raise ValueError("No cation/anion paired data found for t-SNE plot.")
    scatter = build_tsne_scatter(
        paired_records,
        perplexity=args.tsne_perplexity,
        max_iter=args.tsne_max_iter,
    )
    plot_scatter(scatter, scatter_path, args.dpi)

    print(f"Wrote {summary_path}")
    print(f"Wrote {bar_path}")
    print(f"Wrote {scatter_path}")
    print(f"Included CSV files: {len(datasets)}")
    print(f"Scatter points: {len(scatter)}")
    print(f"Unique ion pairs in t-SNE: {scatter[['cation', 'anion']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
