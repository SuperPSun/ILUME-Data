from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.structure_3d_box_features import DEFAULT_OUTPUT, build_dataset
from src.box_features import (
    RDFResult,
    build_reference_topology,
    compute_hydrogen_bond_features,
    compute_rdf,
    compute_structure_factor_features,
    extract_box_features,
    extract_rdf_features,
    map_topology_to_residue,
    minimum_image,
    parse_pdb_box,
    wrapped_center_of_mass,
)


def test_default_output_is_structured_3d_box_csv():
    assert DEFAULT_OUTPUT == Path("data/structured/simulation/3d_box_structured.csv").resolve()


def pdb_atom(serial: int, name: str, residue: str, residue_id: int, xyz: tuple[float, float, float]) -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {name:>4s} {residue:>3s}  {residue_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00            \n"
    )


def write_ammonium_chloride_box(path: Path) -> None:
    lines = [
        "TITLE     synthetic ionic liquid\n",
        "CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1\n",
        "MODEL        1\n",
        pdb_atom(1, "N1", "POS", 1, (1.0, 1.0, 1.0)),
        pdb_atom(2, "H1", "POS", 1, (2.0, 1.0, 1.0)),
        pdb_atom(3, "H2", "POS", 1, (0.0, 1.0, 1.0)),
        pdb_atom(4, "H3", "POS", 1, (1.0, 2.0, 1.0)),
        pdb_atom(5, "H4", "POS", 1, (1.0, 1.0, 2.0)),
        pdb_atom(6, "Cl1", "NEG", 2, (4.0, 1.0, 1.0)),
        "TER\n",
        "ENDMDL\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def write_box_with_late_atom_name_corruption(path: Path) -> None:
    lines = [
        "TITLE     synthetic atom-name rollover\n",
        "CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1\n",
        "MODEL        1\n",
        pdb_atom(1, "N1", "POS", 1, (1.0, 1.0, 1.0)),
        pdb_atom(2, "H1", "POS", 1, (2.0, 1.0, 1.0)),
        pdb_atom(3, "H2", "POS", 1, (0.0, 1.0, 1.0)),
        pdb_atom(4, "H3", "POS", 1, (1.0, 2.0, 1.0)),
        pdb_atom(5, "H4", "POS", 1, (1.0, 1.0, 2.0)),
        pdb_atom(6, "N1", "POS", 2, (11.0, 11.0, 11.0)),
        pdb_atom(7, "H1", "POS", 2, (12.0, 11.0, 11.0)),
        pdb_atom(8, "H2", "POS", 2, (10.0, 11.0, 11.0)),
        pdb_atom(9, "H3", "POS", 2, (11.0, 12.0, 11.0)),
        pdb_atom(10, "H4", "POS", 2, (11.0, 11.0, 12.0)),
        pdb_atom(11, "CPO", "NEG", 3, (4.0, 1.0, 1.0)),
        pdb_atom(12, "CL1", "NEG", 4, (14.0, 11.0, 11.0)),
        "TER\n",
        "ENDMDL\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def test_minimum_image_and_wrapped_center_of_mass():
    delta = minimum_image(np.asarray([9.8, -9.7, 0.2]), np.asarray([10.0, 10.0, 10.0]))
    assert np.allclose(delta, [-0.2, 0.3, 0.2])
    center = wrapped_center_of_mass(
        np.asarray([[9.5, 1.0, 1.0], [10.5, 1.0, 1.0]]),
        np.asarray([1.0, 1.0]),
        np.asarray([10.0, 10.0, 10.0]),
    )
    assert np.allclose(center, [0.0, 1.0, 1.0])


def test_parse_pdb_and_topology_mapping(tmp_path: Path):
    path = tmp_path / "box.pdb"
    write_ammonium_chloride_box(path)
    box = parse_pdb_box(path)
    assert box.lengths.tolist() == [50.0, 50.0, 50.0]
    assert [residue.side for residue in box.residues] == ["cat", "an"]
    assert box.residues[1].elements == ("Cl",)

    row, _curves = extract_box_features(path, "[NH4+]", "[Cl-]")
    assert row["qc_topology_ok"]
    assert row["hb_cat_to_an_strong_count"] == 1.0
    assert row["hb_total_count"] == 1.0
    assert row["qc_polar_heavy_count"] == 2
    assert row["qc_nonpolar_heavy_count"] == 0


def test_topology_mapping_tolerates_a_different_explicit_hydrogen_count(tmp_path: Path):
    path = tmp_path / "box.pdb"
    write_ammonium_chloride_box(path)
    residue = parse_pdb_box(path).residues[0]
    reduced = type(residue)(
        side=residue.side,
        residue_id=residue.residue_id,
        atom_names=residue.atom_names[:-1],
        elements=residue.elements[:-1],
        positions=residue.positions[:-1],
    )
    template = map_topology_to_residue(build_reference_topology("[NH4+]"), reduced)
    assert template.topology_ok
    assert len(template.donor_by_hydrogen) == 3


def test_verified_template_recovers_late_pdb_atom_name_corruption(tmp_path: Path):
    path = tmp_path / "box.pdb"
    write_box_with_late_atom_name_corruption(path)
    row, _curves = extract_box_features(path, "[NH4+]", "[Cl-]")
    assert row["qc_topology_ok"]
    assert "pdb_atom_name_element_mismatch" in row["qc_flags"]
    assert not row["qc_status"] == "failed"


def test_periodic_rdf_is_normalized_for_uniform_points():
    rng = np.random.default_rng(17)
    points = rng.uniform(0.0, 50.0, size=(1500, 3))
    rdf = compute_rdf(points, points, np.asarray([50.0, 50.0, 50.0]), same=True, dr=0.5, r_max=10.0)
    bulk = rdf.g[(rdf.r >= 3.0) & (rdf.r <= 9.0)]
    assert abs(float(np.mean(bulk)) - 1.0) < 0.08


def test_rdf_group_exclusion_removes_intramolecular_pair():
    points = np.asarray([[1.0, 1.0, 1.0], [1.2, 1.0, 1.0], [5.0, 5.0, 5.0]])
    groups = np.asarray([0, 0, 1])
    rdf = compute_rdf(
        points,
        points,
        np.asarray([10.0, 10.0, 10.0]),
        same=True,
        groups_a=groups,
        groups_b=groups,
        dr=0.1,
        r_max=4.0,
    )
    assert np.nanmax(rdf.g[rdf.r < 0.5]) == 0.0


def test_extract_rdf_peak_valley_width_and_area():
    r = np.arange(0.05, 20.0, 0.1)
    g = 1.0 + 3.0 * np.exp(-0.5 * ((r - 5.0) / 0.5) ** 2)
    g += 0.5 * np.exp(-0.5 * ((r - 9.0) / 0.8) ** 2)
    features = extract_rdf_features(RDFResult(r, g, 0.01))
    assert abs(features["peak1_r_A"] - 5.0) < 0.15
    assert 6.0 < features["minimum1_r_A"] < 8.5
    assert 0.8 < features["fwhm_A"] < 1.6
    assert features["excess_area_A"] > 0
    assert features["coordination1"] > 0


def test_hydrogen_bond_boundaries_and_network():
    sites = {
        "donor_h_positions": np.asarray([[1.0, 0.0, 0.0]]),
        "donor_d_positions": np.asarray([[0.0, 0.0, 0.0]]),
        "donor_sides": np.asarray(["cat"], dtype=object),
        "donor_groups": np.asarray([0]),
        "donor_kinds": np.asarray(["strong"], dtype=object),
        "acceptor_positions": np.asarray([[3.5, 0.0, 0.0]]),
        "acceptor_sides": np.asarray(["an"], dtype=object),
        "acceptor_groups": np.asarray([1]),
    }
    result = compute_hydrogen_bond_features(sites, np.asarray([10.0, 10.0, 10.0]), ion_count=2)
    assert result["hb_cat_to_an_strong_count"] == 1.0
    assert result["hb_total_count"] == 1.0
    assert result["hb_loose_total_count"] == 1.0
    assert result["hb_strict_total_count"] == 0.0
    assert result["hb_DHA_angle_mean_deg"] == 180.0
    assert result["hb_network_largest_component_fraction"] == 1.0


def test_polar_partition_handles_charged_heads_and_perfluoro_groups():
    imidazolium = build_reference_topology("CCCCn1cc[nH+]c1")
    assert 0 < int(imidazolium.polar_heavy.sum()) < sum(e != "H" for e in imidazolium.elements)

    pf6 = build_reference_topology("F[P-](F)(F)(F)(F)F")
    assert int(pf6.polar_heavy.sum()) == 7

    tfsi = build_reference_topology("O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F")
    heavy_count = sum(element != "H" for element in tfsi.elements)
    assert 0 < int(tfsi.polar_heavy.sum()) < heavy_count


def test_structure_factor_returns_finite_charge_spectrum():
    rng = np.random.default_rng(4)
    box = np.asarray([20.0, 20.0, 20.0])
    sites = {
        "polar_positions": rng.uniform(0, 20, size=(50, 3)),
        "nonpolar_positions": rng.uniform(0, 20, size=(50, 3)),
        "cat_centers": rng.uniform(0, 20, size=(20, 3)),
        "an_centers": rng.uniform(0, 20, size=(20, 3)),
    }
    _features, curves = compute_structure_factor_features(sites, box)
    assert curves["scc"].shape == (30,)
    assert curves["szz"].shape == (30,)
    assert np.isfinite(curves["szz"]).any()


def test_build_dataset_keeps_missing_pdb_and_resumes(tmp_path: Path):
    box_dir = tmp_path / "raw" / "box"
    box_dir.mkdir(parents=True)
    write_ammonium_chloride_box(box_dir / "mol_present.pdb")
    pd.DataFrame(
        [
            {"mol_id": "mol_present", "cation_smiles": "[NH4+]", "anion_smiles": "[Cl-]", "temperature": 298},
            {"mol_id": "mol_missing", "cation_smiles": "[NH4+]", "anion_smiles": "[Cl-]", "temperature": 298},
        ]
    ).to_csv(box_dir / "mapping.csv", index=False)
    output = tmp_path / "final" / "3d_box.csv"
    audit = tmp_path / "audit"
    first = build_dataset(
        box_dir=box_dir,
        charge_dir=tmp_path / "no-charge",
        output_path=output,
        audit_dir=audit,
        jobs=1,
        batch_size=1,
    )
    second = build_dataset(
        box_dir=box_dir,
        charge_dir=tmp_path / "no-charge",
        output_path=output,
        audit_dir=audit,
        jobs=1,
        batch_size=1,
        resume=True,
    )
    assert len(first) == len(second) == 2
    assert second["mol_id"].tolist() == ["mol_present", "mol_missing"]
    missing = second[second["mol_id"].eq("mol_missing")].iloc[0]
    assert missing["qc_status"] == "failed"
    assert missing["qc_flags"] == "missing_pdb"
    assert (audit / "curves" / "batch_00000.npz").exists()
    assert output.exists()
    assert math.isnan(float(missing["rdf_ca_peak1_r_A"]))
