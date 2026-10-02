"""Publish merged data into final, deriving hydration and excluding selected files."""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHARGE_DATA_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "simulation_data" / "charge_20260514"
)
BUCKETS = ("experiment", "simulation")
EXCLUDED_EXPERIMENT_FILES = (
    "thermal_diffusivity.csv",
    "enthalpy.csv",
    "entropy.csv",
    "heat_capacity_at_vapor_saturation_pressure.csv",
    "enthalpy_of_vaporization_or_sublimation.csv",
    "enthalpy_of_transition_or_fusion.csv",
    "equilibrium_temperature.csv",
    "isobaric_coefficient_of_volume_expansion.csv",
)
EXCLUDED_SIMULATION_FILES = ("3d_box.csv",)
STRUCTURE_MANIFEST_COLUMNS = (
    "mol_id",
    "relative_path",
    "format",
    "size_bytes",
    "sha256",
    "referenced_by_charge",
)


def build_hydration(staged_root: Path) -> None:
    experiment = staged_root / "experiment"
    solvation_path = experiment / "solvation.csv"
    transfer_path = experiment / "transfer.csv"
    if not solvation_path.exists() and not transfer_path.exists():
        return
    if not solvation_path.exists() or not transfer_path.exists():
        raise ValueError("Hydration requires both solvation.csv and transfer.csv")

    keys = ["cation", "anion", "solute", "temperature_K"]
    frames = []
    for path, label in (
        (solvation_path, "solvation_kcal/mol"),
        (transfer_path, "transfer_kcal/mol"),
    ):
        frame = pd.read_csv(path)
        required = [*keys, label, "source_list"]
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        frame = frame[required].copy()
        for column in keys[:3]:
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"{path.name}: invalid {column}")
        for column in ("temperature_K", label):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not frame[column].map(math.isfinite).all():
                raise ValueError(f"{path.name}: non-finite {column}")
        frames.append(frame)

    solvation, transfer = frames
    candidates = transfer.merge(
        solvation, on=keys, how="left", suffixes=("_transfer", "_solvation"), indicator=True
    )
    # The merge suffixes source_list; restore the original transfer schema for auditing.
    unmatched = candidates.loc[
        candidates["_merge"].eq("left_only"),
        [*keys, "transfer_kcal/mol", "source_list_transfer"],
    ].rename(columns={"source_list_transfer": "source_list"})
    paired = candidates.loc[candidates["_merge"].eq("both")].drop(columns="_merge").copy()
    # Existing transfer labels use the sign convention hydration = solvation + transfer.
    paired["hydration_kcal/mol"] = paired["solvation_kcal/mol"] + paired["transfer_kcal/mol"]
    if not paired["hydration_kcal/mol"].map(math.isfinite).all():
        raise ValueError("Non-finite derived hydration")

    def combine_sources(values: pd.Series) -> str:
        return "; ".join(sorted({
            token.strip()
            for value in values.dropna()
            for token in str(value).split(";")
            if token.strip()
        }))

    paired["source_list"] = (
        paired["source_list_solvation"].fillna("") + "; "
        + paired["source_list_transfer"].fillna("")
    )
    grouped = paired.groupby(["solute", "temperature_K"], sort=True)
    summary = grouped["hydration_kcal/mol"].agg(
        candidate_count="count", minimum_kcal_mol="min",
        maximum_kcal_mol="max", median_kcal_mol="median",
    ).reset_index()
    hydration = grouped.agg({"hydration_kcal/mol": "median", "source_list": combine_sources}).reset_index()
    hydration.to_csv(experiment / "hydration.csv", index=False)
    audit = staged_root / "_audit"
    audit.mkdir(exist_ok=True)
    paired.drop(columns="source_list").to_csv(audit / "hydration_pairs.csv", index=False)
    summary.to_csv(audit / "hydration_summary.csv", index=False)
    unmatched.to_csv(audit / "hydration_unmatched_transfer.csv", index=False)
    transfer_path.unlink()


def remove_charge_mapping(staged_root: Path) -> None:
    charge_structure_root = staged_root / "simulation" / "charge_20260514"
    (charge_structure_root / "mapping.csv").unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_structure_manifest(staged_root: Path) -> None:
    simulation_root = staged_root / "simulation"
    structure_root = simulation_root / "charge_20260514"
    charge = pd.read_csv(simulation_root / "charge.csv", usecols=["mol_id"])
    referenced = set(charge["mol_id"].astype(str))
    rows = []
    for path in sorted(structure_root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in {".mol", ".mol2"}:
            continue
        rows.append(
            {
                "mol_id": path.stem,
                "relative_path": path.relative_to(structure_root).as_posix(),
                "format": path.suffix.lower().removeprefix("."),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "referenced_by_charge": path.stem in referenced,
            }
        )
    pd.DataFrame(rows, columns=STRUCTURE_MANIFEST_COLUMNS).to_csv(
        structure_root / "structure_manifest.csv",
        index=False,
    )


def build_final_data(
    input_root: Path,
    output_root: Path,
    charge_data_root: Path = CHARGE_DATA_ROOT,
) -> None:
    """Rebuild ``output_root`` from merged data while omitting excluded files."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    charge_data_root = Path(charge_data_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output_root.parent) as temporary_dir:
        staged_root = Path(temporary_dir) / output_root.name
        for bucket in BUCKETS:
            shutil.copytree(input_root / bucket, staged_root / bucket)
        shutil.copytree(
            charge_data_root,
            staged_root / "simulation" / charge_data_root.name,
        )
        for filename in EXCLUDED_EXPERIMENT_FILES:
            (staged_root / "experiment" / filename).unlink(missing_ok=True)
        for filename in EXCLUDED_SIMULATION_FILES:
            (staged_root / "simulation" / filename).unlink(missing_ok=True)
        remove_charge_mapping(staged_root)
        merged_audit = input_root / "_audit"
        if merged_audit.exists():
            shutil.copytree(merged_audit, staged_root / "_audit")
        build_hydration(staged_root)
        write_structure_manifest(staged_root)

        if output_root.exists():
            shutil.rmtree(output_root)
        staged_root.rename(output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "merged")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "final")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_final_data(args.input_root, args.output_root)


if __name__ == "__main__":
    main()
