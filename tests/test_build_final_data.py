from pathlib import Path
import hashlib

import pandas as pd

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
    ):
        assert not (final_root / "experiment" / filename).exists()
