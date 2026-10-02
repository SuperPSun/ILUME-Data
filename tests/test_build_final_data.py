from pathlib import Path
import hashlib

import pandas as pd
import pytest

from scripts.build_final_data import build_final_data


def test_build_final_data_copies_buckets_and_excludes_requested_experiment_properties(tmp_path: Path):
    merged_root = tmp_path / "merged"
    experiment = merged_root / "experiment"
    simulation = merged_root / "simulation"
    experiment.mkdir(parents=True)
    simulation.mkdir()
    merged_audit = merged_root / "_audit"
    merged_audit.mkdir()
    gap_audit = merged_audit / "orbital_gap_consistency_summary.csv"
    gap_audit.write_text("source_file,checked_rows\norbital.csv,2\n")

    retained_experiment = experiment / "density.csv"
    retained_experiment.write_bytes(b"density,1\\n")
    for filename in (
        "thermal_diffusivity.csv",
        "enthalpy.csv",
        "entropy.csv",
        "heat_capacity_at_vapor_saturation_pressure.csv",
        "enthalpy_of_vaporization_or_sublimation.csv",
        "enthalpy_of_transition_or_fusion.csv",
        "equilibrium_temperature.csv",
        "isobaric_coefficient_of_volume_expansion.csv",
    ):
        (experiment / filename).write_bytes(b"excluded,1\\n")
    retained_simulation = simulation / "heat_of_vaporization.csv"
    retained_simulation.write_bytes(b"heat_of_vaporization,1\\n")
    pd.DataFrame(
        {
            "mol_id": ["mol_0000000", "mol_0022092"],
            "SMILES": ["CCO", "CC"],
            "charge": [0, -1],
            "source_list": ["simulation", "simulation"],
        }
    ).to_csv(simulation / "charge.csv", index=False)
    box_features = simulation / "3d_box.csv"
    box_features.write_bytes(b"mol_id,rdf_ca_peak1_r_A\\nmol_1,4.1\\n")
    charge_data_root = tmp_path / "raw" / "simulation_data" / "charge_20260514"
    charge_data_root.mkdir(parents=True)
    charge_file = charge_data_root / "mol_0000000.mol2"
    charge_file.write_bytes(b"charge data")
    retained_mol = charge_data_root / "mol_0022092.mol"
    retained_mol.write_bytes(b"retained mol")
    retained_mol2 = charge_data_root / "mol_0022092.mol2"
    retained_mol2.write_bytes(b"retained mol2")
    orphan = charge_data_root / "mol_orphan.mol2"
    orphan.write_bytes(b"orphan")
    mapping = charge_data_root / "mapping.csv"
    mapping.write_text("mol_id,smiles,charge\\nmol_0000000,CCO,0\\nmol_0022092,CC,-1\\n")

    final_root = tmp_path / "final"
    stale_file = final_root / "experiment" / "stale.csv"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_bytes(b"stale,1\\n")

    build_final_data(merged_root, final_root, charge_data_root)

    assert (final_root / "experiment" / "density.csv").read_bytes() == retained_experiment.read_bytes()
    assert (final_root / "simulation" / "heat_of_vaporization.csv").read_bytes() == retained_simulation.read_bytes()
    assert not (final_root / "simulation" / "3d_box.csv").exists()
    assert (
        final_root / "simulation" / "charge_20260514" / "mol_0000000.mol2"
    ).read_bytes() == charge_file.read_bytes()
    final_charge = pd.read_csv(final_root / "simulation" / "charge.csv")
    assert final_charge["mol_id"].tolist() == ["mol_0000000", "mol_0022092"]
    assert (final_root / "simulation" / "charge_20260514" / retained_mol.name).exists()
    assert (final_root / "simulation" / "charge_20260514" / retained_mol2.name).exists()
    assert not (final_root / "simulation" / "charge_20260514" / "mapping.csv").exists()
    manifest_path = (
        final_root / "simulation" / "charge_20260514" / "structure_manifest.csv"
    )
    structure_manifest = pd.read_csv(manifest_path)
    assert list(structure_manifest.columns) == [
        "mol_id",
        "relative_path",
        "format",
        "size_bytes",
        "sha256",
        "referenced_by_charge",
    ]
    assert structure_manifest["relative_path"].tolist() == [
        "mol_0000000.mol2",
        "mol_0022092.mol",
        "mol_0022092.mol2",
        "mol_orphan.mol2",
    ]
    assert structure_manifest["referenced_by_charge"].tolist() == [
        True,
        True,
        True,
        False,
    ]
    first = structure_manifest.iloc[0]
    assert first["size_bytes"] == len(b"charge data")
    assert first["sha256"] == hashlib.sha256(b"charge data").hexdigest()
    assert not structure_manifest["relative_path"].eq("structure_manifest.csv").any()
    assert (
        final_root / "_audit" / "orbital_gap_consistency_summary.csv"
    ).read_bytes() == gap_audit.read_bytes()
    first_manifest = manifest_path.read_bytes()
    (
        merged_root
        / "experiment"
        / "isobaric_coefficient_of_volume_expansion.csv"
    ).unlink()
    build_final_data(merged_root, final_root, charge_data_root)
    assert manifest_path.read_bytes() == first_manifest
    assert not stale_file.exists()
    for filename in (
        "thermal_diffusivity.csv",
        "enthalpy.csv",
        "entropy.csv",
        "heat_capacity_at_vapor_saturation_pressure.csv",
        "enthalpy_of_vaporization_or_sublimation.csv",
        "enthalpy_of_transition_or_fusion.csv",
        "equilibrium_temperature.csv",
        "isobaric_coefficient_of_volume_expansion.csv",
    ):
        assert not (final_root / "experiment" / filename).exists()


@pytest.fixture
def hydration_inputs(tmp_path):
    merged = tmp_path / "merged"
    experiment = merged / "experiment"
    experiment.mkdir(parents=True)
    simulation = merged / "simulation"
    simulation.mkdir()
    pd.DataFrame({"mol_id": []}).to_csv(simulation / "charge.csv", index=False)
    charge = tmp_path / "charge_20260514"
    charge.mkdir()
    columns = ["cation", "anion", "solute", "temperature_K", "solvation_kcal/mol", "source_list"]
    pd.DataFrame([
        ["A", "X", "CC", 298.15, -5, "alpha; shared"],
        ["A", "X", "CC", 298.15, -3, "beta"],
        ["B", "Y", "CC", 298.15, -1, "gamma"],
        ["A", "X", "CC", 310, -9, "hot"],
        ["A", "X", "CO", 298.15, -7, "other"],
    ], columns=columns).to_csv(experiment / "solvation.csv", index=False)
    columns[4] = "transfer_kcal/mol"
    pd.DataFrame([
        ["A", "X", "CC", 298.15, 2, "shared; delta"],
        ["B", "Y", "CC", 298.15, 9, "epsilon"],
        ["A", "X", "CC", 310, 1, "hot"],
        ["A", "X", "CO", 298.15, 3, "other"],
        ["A", "X", "CO", 310, 3, "unmatched"],
    ], columns=columns).to_csv(experiment / "transfer.csv", index=False)
    (experiment / "transfer_organic.csv").write_text("preserved\n")
    return merged, tmp_path / "final", charge


def test_hydration_pairing_median_and_audit(hydration_inputs):
    merged, final, charge = hydration_inputs
    original = (merged / "experiment" / "solvation.csv").read_bytes()
    build_final_data(merged, final, charge)
    hydration = pd.read_csv(final / "experiment" / "hydration.csv")
    assert hydration.columns.tolist() == ["solute", "temperature_K", "hydration_kcal/mol", "source_list"]
    assert not hydration.duplicated(["solute", "temperature_K"]).any()
    assert hydration["hydration_kcal/mol"].tolist() == [-1, -8, -4]
    assert hydration.iloc[0]["source_list"] == "alpha; beta; delta; epsilon; gamma; shared"
    pairs = pd.read_csv(final / "_audit" / "hydration_pairs.csv")
    assert len(pairs) == 5
    assert pairs["hydration_kcal/mol"].tolist() == [-3, -1, 8, -8, -4]
    summary = pd.read_csv(final / "_audit" / "hydration_summary.csv")
    assert summary.iloc[0]["candidate_count"] == 3
    assert summary.iloc[0]["minimum_kcal_mol"] == -3
    assert summary.iloc[0]["maximum_kcal_mol"] == 8
    assert summary.iloc[0]["median_kcal_mol"] == -1
    unmatched = pd.read_csv(final / "_audit" / "hydration_unmatched_transfer.csv")
    assert unmatched[["solute", "temperature_K", "source_list"]].values.tolist() == [["CO", 310, "unmatched"]]
    assert not (final / "experiment" / "transfer.csv").exists()
    assert (merged / "experiment" / "transfer.csv").exists()
    assert (final / "experiment" / "solvation.csv").read_bytes() == original
    assert (final / "experiment" / "transfer_organic.csv").read_text() == "preserved\n"
    first = (final / "experiment" / "hydration.csv").read_bytes()
    build_final_data(merged, final, charge)
    assert (final / "experiment" / "hydration.csv").read_bytes() == first


@pytest.mark.parametrize("invalid", ["missing_transfer", "missing_solvation", "missing_field", "text", "nan", "infinite", "temperature", "identity"])
def test_invalid_hydration_preserves_existing_final(hydration_inputs, invalid):
    merged, final, charge = hydration_inputs
    experiment = merged / "experiment"
    if invalid.startswith("missing_") and invalid != "missing_field":
        (experiment / (invalid.removeprefix("missing_") + ".csv")).unlink()
    else:
        path = experiment / "solvation.csv"
        frame = pd.read_csv(path)
        if invalid == "missing_field":
            frame = frame.drop(columns="solute")
        elif invalid == "identity":
            frame.loc[0, "solute"] = " "
        elif invalid == "temperature":
            frame.loc[0, "temperature_K"] = float("inf")
        else:
            frame["solvation_kcal/mol"] = frame["solvation_kcal/mol"].astype(object)
            frame.loc[0, "solvation_kcal/mol"] = {"text": "invalid", "nan": float("nan"), "infinite": float("inf")}[invalid]
        frame.to_csv(path, index=False)
    final.mkdir()
    marker = final / "existing.csv"
    marker.write_bytes(b"original output")
    with pytest.raises(ValueError):
        build_final_data(merged, final, charge)
    assert list(final.iterdir()) == [marker]
    assert marker.read_bytes() == b"original output"


def test_hydration_without_matches_has_empty_output(hydration_inputs):
    merged, final, charge = hydration_inputs
    path = merged / "experiment" / "transfer.csv"
    frame = pd.read_csv(path)
    frame["temperature_K"] = 400.0
    frame.to_csv(path, index=False)
    build_final_data(merged, final, charge)
    assert pd.read_csv(final / "experiment" / "hydration.csv").empty
    assert pd.read_csv(final / "_audit" / "hydration_summary.csv").empty
    assert len(pd.read_csv(final / "_audit" / "hydration_unmatched_transfer.csv")) == 5
