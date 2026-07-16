"""Reproducible CSV-only quality checks for ``data/final``.

The audit intentionally excludes MOL2 files. It reports evidence without
modifying the source datasets.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


IDENTIFIER_COLUMNS = ("mol_id", "cation", "anion", "solute", "solvent", "smiles", "SMILES")
CONDITION_COLUMNS = ("temperature_K", "pressure_kPa", "frequency_MHz", "wavelength_nm", "phase")
METADATA_COLUMNS = ("standard_state_note", "source_list")
GRAIN_COLUMNS = (*IDENTIFIER_COLUMNS, *CONDITION_COLUMNS, "standard_state_note")


def csv_paths(final_root: Path) -> list[Path]:
    return sorted(Path(final_root).rglob("*.csv"))


def label_columns(frame: pd.DataFrame) -> list[str]:
    excluded = set((*GRAIN_COLUMNS, *METADATA_COLUMNS))
    return [column for column in frame.columns if column not in excluded]


def profile_files(final_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in csv_paths(final_root):
        frame = pd.read_csv(path, low_memory=False)
        numeric = frame.select_dtypes(include="number")
        nonfinite = int((~np.isfinite(numeric)).sum().sum() - numeric.isna().sum().sum())
        labels = label_columns(frame)
        rows.append(
            {
                "file": str(path.relative_to(final_root)),
                "rows": len(frame),
                "columns": len(frame.columns),
                "exact_duplicate_rows": int(frame.duplicated().sum()),
                "missing_cells": int(frame.isna().sum().sum()),
                "missing_label_cells": int(frame[labels].isna().sum().sum()) if labels else 0,
                "nonfinite_numeric_cells": nonfinite,
            }
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def formal_charge(smiles: str) -> int | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())


def ion_role_summary(final_root: Path) -> pd.DataFrame:
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    rows: list[dict[str, object]] = []
    for path in sorted((Path(final_root) / "experiment").glob("*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        if not {"cation", "anion"}.issubset(frame.columns):
            continue
        cation_charge = frame["cation"].map(formal_charge)
        anion_charge = frame["anion"].map(formal_charge)
        same_ion = frame["cation"].eq(frame["anion"])
        wrong_role = cation_charge.le(0) | anion_charge.ge(0)
        if same_ion.any() or wrong_role.any():
            rows.append(
                {
                    "file": str(path.relative_to(final_root)),
                    "rows": len(frame),
                    "cation_equals_anion": int(same_ion.sum()),
                    "invalid_ion_role_rows": int(wrong_role.sum()),
                    "invalid_ion_role_rate": float(wrong_role.mean()),
                }
            )
    return pd.DataFrame(rows)


def grain_conflict_summary(final_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in csv_paths(final_root):
        frame = pd.read_csv(path, low_memory=False)
        labels = label_columns(frame)
        keys = [column for column in GRAIN_COLUMNS if column in frame.columns]
        if len(labels) != 1 or not keys:
            continue
        label = labels[0]
        grouped = frame.groupby(keys, dropna=False)[label].agg(["size", "min", "max"])
        conflicts = grouped[grouped["size"].gt(1)]
        if conflicts.empty:
            continue
        spread = conflicts["max"] - conflicts["min"]
        rows.append(
            {
                "file": str(path.relative_to(final_root)),
                "label": label,
                "rows": len(frame),
                "conflict_groups": len(conflicts),
                "affected_rows": int(conflicts["size"].sum()),
                "affected_rate": float(conflicts["size"].sum() / len(frame)),
                "median_range": float(spread.median()),
                "p95_range": float(spread.quantile(0.95)),
                "max_range": float(spread.max()),
                "max_group_size": int(conflicts["size"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("affected_rate", ascending=False).reset_index(drop=True)


def pressure_missingness(final_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((Path(final_root) / "experiment").glob("*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        if "pressure_kPa" not in frame.columns or not frame["pressure_kPa"].isna().any():
            continue
        missing = int(frame["pressure_kPa"].isna().sum())
        rows.append(
            {
                "file": str(path.relative_to(final_root)),
                "rows": len(frame),
                "missing_pressure_rows": missing,
                "missing_pressure_rate": missing / len(frame),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_pressure_rate", ascending=False).reset_index(drop=True)


def zero_loss_evidence(project_root: Path) -> pd.DataFrame:
    project_root = Path(project_root)
    raw = project_root / "data" / "raw" / "simulation_data"
    final = project_root / "data" / "final" / "simulation"
    qm_raw = pd.read_csv(raw / "QM_elec_HF.csv", low_memory=False)
    qm_final = pd.read_csv(final / "simulated_qm_elec_hf.csv", low_memory=False)
    mapping = pd.read_csv(final / "charge_20260514" / "mapping.csv", low_memory=False)
    charge = pd.read_csv(final / "charge.csv", low_memory=False)
    transfer_raw = pd.read_csv(raw / "combi_qm_solv.csv", low_memory=False)
    transfer_final = pd.read_csv(final / "transfer_organic.csv", low_memory=False)

    rows = [
        {
            "field": "charge",
            "raw_zero_values": int(mapping["charge"].eq(0).sum()),
            "final_missing_or_dropped": int(mapping["mol_id"].nunique() - charge["mol_id"].nunique()),
            "final_context": "neutral rows absent from simulation/charge.csv",
        },
        {
            "field": "ESP_pos_frac",
            "raw_zero_values": int(qm_raw["ESP_pos_frac"].eq(0).sum()),
            "final_missing_or_dropped": int(qm_final["ESP_pos_frac"].isna().sum()),
            "final_context": "zero fractions became null after rejected/duplicate rows were removed",
        },
        {
            "field": "Dipole",
            "raw_zero_values": int(qm_raw["Dipole"].eq(0).sum()),
            "final_missing_or_dropped": int(qm_final["Dipole"].isna().sum()),
            "final_context": "zero dipoles became null after rejected/duplicate rows were removed",
        },
        {
            "field": "transfer_organic_kcal/mol",
            "raw_zero_values": int(transfer_raw["solv"].eq(0).sum()),
            "final_missing_or_dropped": int(len(transfer_raw) - len(transfer_final)),
            "final_context": "zero labels were rejected as missing",
        },
    ]
    return pd.DataFrame(rows)


def qm_checks(final_root: Path) -> dict[str, object]:
    frame = pd.read_csv(Path(final_root) / "simulation" / "simulated_qm_elec_hf.csv")
    group_sizes = frame.groupby("SMILES").size()
    duplicate_groups = group_sizes[group_sizes.gt(1)]
    gap_spread = frame.groupby("SMILES")["gap_eV"].agg(lambda values: values.max() - values.min())
    return {
        "rows": len(frame),
        "unique_smiles": int(frame["SMILES"].nunique()),
        "duplicate_smiles_groups": len(duplicate_groups),
        "affected_duplicate_rows": int(duplicate_groups.sum()),
        "max_rows_per_smiles": int(duplicate_groups.max()),
        "median_gap_spread_eV": float(gap_spread.loc[duplicate_groups.index].median()),
        "max_gap_spread_eV": float(gap_spread.loc[duplicate_groups.index].max()),
        "negative_gap_rows": int(frame["gap_eV"].lt(0).sum()),
        "missing_label_cells": int(frame.drop(columns=["SMILES", "source_list"]).isna().sum().sum()),
    }


def mapping_checks(final_root: Path) -> dict[str, object]:
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    mapping = pd.read_csv(Path(final_root) / "simulation" / "charge_20260514" / "mapping.csv")
    parsed = mapping["smiles"].map(Chem.MolFromSmiles)
    invalid = parsed.isna()
    canonical = parsed.map(
        lambda molecule: Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if molecule is not None
        else None
    )
    return {
        "rows": len(mapping),
        "unique_mol_id": int(mapping["mol_id"].nunique()),
        "invalid_smiles_rows": int(invalid.sum()),
        "noncanonical_smiles_rows": int((canonical.notna() & canonical.ne(mapping["smiles"])).sum()),
        "noncanonical_unique_smiles": int(mapping.loc[canonical.notna() & canonical.ne(mapping["smiles"]), "smiles"].nunique()),
    }


def run_audit(project_root: Path) -> dict[str, object]:
    project_root = Path(project_root).resolve()
    final_root = project_root / "data" / "final"
    profiles = profile_files(final_root)
    return {
        "profiles": profiles,
        "ion_roles": ion_role_summary(final_root),
        "grain_conflicts": grain_conflict_summary(final_root),
        "pressure_missingness": pressure_missingness(final_root),
        "zero_loss": zero_loss_evidence(project_root),
        "qm": qm_checks(final_root),
        "mapping": mapping_checks(final_root),
        "totals": {
            "csv_files": len(profiles),
            "rows": int(profiles["rows"].sum()),
            "exact_duplicate_rows": int(profiles["exact_duplicate_rows"].sum()),
            "nonfinite_numeric_cells": int(profiles["nonfinite_numeric_cells"].sum()),
        },
    }
