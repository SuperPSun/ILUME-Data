from pathlib import Path

import pandas as pd
import pytest

from scripts.merge_data import fixed_h_inchikey, merge_data, property_slug


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


DEFAULT_PRESSURE_PROPERTIES = (
    ("density_g/cm^3", "density.csv"),
    ("electrical_conductivity_S/m_log10", "electrical_conductivity.csv"),
    ("heat_capacity_J/mol/K", "heat_capacity.csv"),
    ("refractive_index_unitless", "refractive_index.csv"),
    ("thermal_conductivity_W/m/K", "thermal_conductivity.csv"),
    ("viscosity_mPa*s_log10", "viscosity.csv"),
)


@pytest.mark.parametrize(("label", "output_name"), DEFAULT_PRESSURE_PROPERTIES)
@pytest.mark.parametrize("missing_kind", ("column_absent", "cell_missing"))
def test_experiment_properties_default_missing_pressure(
    tmp_path: Path,
    label: str,
    output_name: str,
    missing_kind: str,
):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    row = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
        label: 1.2,
    }
    if missing_kind == "cell_missing":
        row["pressure_kPa"] = None
    write_csv(input_root / "AIonopedia" / output_name, [row])

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / output_name)
    assert out["pressure_kPa"].tolist() == [101.325]


def test_default_pressure_merges_with_explicit_standard_pressure_only(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    base = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
        "density_g/cm^3": 1.2,
    }
    write_csv(input_root / "AIonopedia" / "density.csv", [base])
    write_csv(
        input_root / "ILThermo" / "density.csv",
        [{**base, "pressure_kPa": 101.325}],
    )
    write_csv(
        input_root / "ILBERT" / "density.csv",
        [{**base, "pressure_kPa": 100.0}],
    )

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "density.csv")
    assert out["pressure_kPa"].tolist() == [100.0, 101.325]
    assert out.loc[out["pressure_kPa"].eq(100.0), "source_list"].item() == "ILBERT"
    assert (
        out.loc[out["pressure_kPa"].eq(101.325), "source_list"].item()
        == "AIonopedia; ILThermo"
    )


@pytest.mark.parametrize(
    ("label", "output_name"),
    (
        ("density_g/cm^3", "density.csv"),
        ("heat_capacity_J/mol/K", "heat_capacity.csv"),
    ),
)
def test_simulation_properties_do_not_receive_default_pressure(
    tmp_path: Path,
    label: str,
    output_name: str,
):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    write_csv(
        input_root / "simulation" / output_name,
        [
            {
                "cation": "CC[n+]1ccn(C)c1",
                "anion": "F[B-](F)(F)F",
                "temperature_K": 298.15,
                label: 1.2,
            }
        ],
    )

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "simulation" / output_name)
    assert "pressure_kPa" not in out.columns


def test_property_slug_uses_stable_filename_rules():
    assert property_slug("density_g/cm^3") == "density"
    assert property_slug("viscosity_mPa*s_log10") == "viscosity"
    assert property_slug("heat_capacity_J/mol/K") == "heat_capacity"
    assert property_slug("pEC50") == "pec50"
    assert property_slug("partition_log10") == "partition"
    assert property_slug("HOMO_eV") == "homo"
    assert property_slug("anion_HOMO_eV") == "anion_homo"
    assert property_slug("cation_LUMO_eV") == "cation_lumo"
    assert property_slug("x_CO2_unitless") == "x_co2"
    assert property_slug("simulated_QM_elec_HF") == "simulated_qm_elec_hf"


def test_fixed_h_identity_merges_resonance_but_not_tautomers():
    assert fixed_h_inchikey("CCCC[n+]1ccn(C)c1") == fixed_h_inchikey(
        "CCCCn1cc[n+](C)c1"
    )
    assert fixed_h_inchikey("O=c1cccc[nH]1") != fixed_h_inchikey(
        "Oc1ccccn1"
    )


def test_merge_normalizes_equivalent_smiles_before_property_aggregation(
    tmp_path: Path,
):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    first = {
        "cation": "CCCC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
        "density_g/cm^3": 1.2,
    }
    equivalent = {
        **first,
        "cation": "CCCCn1cc[n+](C)c1",
    }
    write_csv(
        input_root / "AIonopedia" / "density.csv",
        [first],
    )
    write_csv(
        input_root / "ILBERT" / "density.csv",
        [
            equivalent,
            {
                **equivalent,
                "temperature_K": 308.15,
                "density_g/cm^3": 1.1,
            },
        ],
    )

    merge_data(input_root, output_root)

    density = pd.read_csv(output_root / "experiment" / "density.csv")
    assert len(density) == 2
    assert density["cation"].nunique() == 1
    assert density.loc[
        density["temperature_K"].eq(298.15),
        "source_list",
    ].item() == "AIonopedia; ILBERT"

    audit = pd.read_csv(
        output_root / "chemical_identity_equivalences.csv"
    )
    row = audit[audit["representative_smiles"].eq(density["cation"].iloc[0])]
    assert len(row) == 1
    assert set(row.iloc[0]["equivalent_smiles_list"].split("; ")) == {
        "CCCC[n+]1ccn(C)c1",
        "CCCCn1cc[n+](C)c1",
    }
    assert row.iloc[0]["occurrence_count"] == 3
    assert row.iloc[0]["source_list"] == "AIonopedia; ILBERT"


def test_qm_median_groups_fixed_h_equivalent_smiles(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    write_csv(
        input_root / "simulation" / "simulated_QM_elec_HF_structured.csv",
        [
            {
                "SMILES": "CCCC[n+]1ccn(C)c1",
                "ESP_max": 1.0,
                "gap_eV": 2.0,
            },
            {
                "SMILES": "CCCCn1cc[n+](C)c1",
                "ESP_max": 3.0,
                "gap_eV": 4.0,
            },
        ],
    )

    merge_data(input_root, output_root)

    qm = pd.read_csv(
        output_root / "simulation" / "simulated_qm_elec_hf.csv"
    )
    assert len(qm) == 1
    assert qm.loc[0, "ESP_max"] == 2.0
    assert qm.loc[0, "gap_eV"] == 3.0


def test_merge_rejects_invalid_chemical_identity_with_context(
    tmp_path: Path,
):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    write_csv(
        input_root / "ILBERT" / "density.csv",
        [
            {
                "cation": "not-a-smiles",
                "anion": "[Cl-]",
                "density_g/cm^3": 1.0,
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"ILBERT/density\.csv, column cation",
    ):
        merge_data(input_root, output_root)


def test_experiment_sources_merge_by_property_and_aggregate_identical_records(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    duplicate_row = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
        "density_g/cm^3": 1.2,
    }
    write_csv(input_root / "AIonopedia" / "AIonopedia_density_structured.csv", [duplicate_row])
    write_csv(input_root / "ILBERT" / "ILBERT_density_structured.csv", [duplicate_row])
    write_csv(input_root / "ILThermo" / "ilt_density_structured.csv", [{**duplicate_row, "phase": "Liquid"}])
    write_csv(input_root / "after_AIonopedia" / "after_AIonopedia_density_structured.csv", [{**duplicate_row, "density_g/cm^3": 1.3}])

    results = merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "density.csv")
    assert len(out) == 2
    assert list(out.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "pressure_kPa",
        "phase",
        "density_g/cm^3",
        "source_list",
    ]
    assert out["pressure_kPa"].eq(101.325).all()
    aggregated = out[out["density_g/cm^3"] == 1.2].iloc[0]
    assert aggregated["phase"] == "Liquid"
    assert aggregated["source_list"] == "AIonopedia; ILBERT; ILThermo"
    assert set(out["density_g/cm^3"]) == {1.2, 1.3}
    assert not (output_root / "AIonopedia").exists()
    assert not (output_root / "ILBERT").exists()
    assert not (output_root / "ILThermo").exists()
    assert not (output_root / "after_AIonopedia").exists()
    assert (output_root / "experiment").is_dir()
    assert (output_root / "simulation").is_dir()
    assert any(result["bucket"] == "experiment" and result["property_label"] == "density_g/cm^3" for result in results)


def test_after_aionopedia_part_folds_precision_difference_and_preserves_material_difference(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    duplicate = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "solute": "CCO",
        "temperature_K": 298.15,
    }
    revised = {
        "cation": "CCCC[n+]1ccn(C)c1",
        "anion": "N#C[N-]C#N",
        "solute": "C1COCCO1",
        "temperature_K": 298.15,
    }
    write_csv(
        input_root / "AIonopedia" / "AIonopedia_transfer_structured.csv",
        [
            {**duplicate, "transfer_kcal/mol": 1.25},
            {**revised, "transfer_kcal/mol": 2.0},
        ],
    )
    write_csv(
        input_root / "after_AIonopedia" / "after_AIonopedia_part_structured.csv",
        [
            {**duplicate, "partition_log10": 1.2500005},
            {**revised, "partition_log10": 2.5},
        ],
    )

    merge_data(input_root, output_root)

    transfer = pd.read_csv(output_root / "experiment" / "transfer.csv")
    assert len(transfer) == 3
    assert set(transfer["transfer_kcal/mol"]) == {1.2500005, 2.0, 2.5}
    assert transfer.loc[transfer["transfer_kcal/mol"].eq(1.2500005), "source_list"].item() == (
        "AIonopedia; after_AIonopedia"
    )
    assert transfer.loc[transfer["transfer_kcal/mol"].eq(2.0), "source_list"].item() == "AIonopedia"
    assert transfer.loc[transfer["transfer_kcal/mol"].eq(2.5), "source_list"].item() == "after_AIonopedia"
    assert not (output_root / "experiment" / "partition.csv").exists()

    manifest = pd.read_csv(output_root / "merged_manifest.csv")
    row = manifest[manifest["property_label"].eq("transfer_kcal/mol")].iloc[0]
    assert row["input_files"] == (
        "AIonopedia/AIonopedia_transfer_structured.csv; "
        "after_AIonopedia/after_AIonopedia_part_structured.csv"
    )
    assert row["input_rows"] == 4
    assert row["output_rows"] == 3


def test_scalar_properties_fold_close_values_without_crossing_conditions_or_tolerance(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    cations = {
        "boundary": "C[N+](C)(C)C",
        "above": "CC[N+](C)(C)C",
        "condition": "CCC[N+](C)(C)C",
        "artifact": "CCCC[N+](C)(C)C",
        "chain": "CCCCC[N+](C)(C)C",
    }

    def row(case: str, value: float, temperature: float = 298.15) -> dict[str, object]:
        return {
            "cation": cations[case],
            "anion": "[Cl-]",
            "temperature_K": temperature,
            "density_g/cm^3": value,
        }

    write_csv(
        input_root / "AIonopedia" / "AIonopedia_density_structured.csv",
        [
            row("boundary", 1.25),
            row("above", 1.25),
            row("condition", 1.25),
            row("artifact", 1.1206),
            row("chain", 1.0),
        ],
    )
    write_csv(
        input_root / "ILBERT" / "ILBERT_density_structured.csv",
        [
            row("boundary", 1.2500005),
            row("above", 1.2500006),
            row("condition", 1.2500004, temperature=308.15),
            row("chain", 1.0000004),
        ],
    )
    write_csv(
        input_root / "after_AIonopedia" / "after_AIonopedia_density_structured.csv",
        [
            row("artifact", 1.1205999999999998),
            row("chain", 1.0000008),
        ],
    )

    merge_data(input_root, output_root)

    density = pd.read_csv(output_root / "experiment" / "density.csv")
    assert len(density) == 8

    boundary = density[density["cation"].eq(cations["boundary"])]
    assert boundary["density_g/cm^3"].tolist() == [1.2500005]
    assert boundary["source_list"].item() == "AIonopedia; ILBERT"

    above = density[density["cation"].eq(cations["above"])]
    assert set(above["density_g/cm^3"]) == {1.25, 1.2500006}

    condition = density[density["cation"].eq(cations["condition"])]
    assert set(condition["temperature_K"]) == {298.15, 308.15}

    artifact = density[density["cation"].eq(cations["artifact"])]
    assert artifact["density_g/cm^3"].tolist() == [1.1206]
    assert artifact["source_list"].item() == "AIonopedia; after_AIonopedia"

    chain = density[density["cation"].eq(cations["chain"])]
    assert set(chain["density_g/cm^3"]) == {1.0000004, 1.0000008}
    assert set(chain["source_list"]) == {"AIonopedia; ILBERT", "after_AIonopedia"}


def test_close_value_exclusions_are_saved_for_audit_and_stale_files_are_removed(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    base = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
    }
    aionopedia_path = input_root / "AIonopedia" / "AIonopedia_density_structured.csv"
    ilbert_path = input_root / "ILBERT" / "ILBERT_density_structured.csv"
    write_csv(aionopedia_path, [{**base, "density_g/cm^3": 1.25}])
    write_csv(ilbert_path, [{**base, "density_g/cm^3": 1.2500005}])

    merge_data(input_root, output_root)

    rejected_path = (
        output_root
        / "rejected_rows"
        / "experiment"
        / "density_approximate_values_rejected.csv"
    )
    rejected = pd.read_csv(rejected_path)
    assert list(rejected.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "pressure_kPa",
        "density_g/cm^3",
        "retained_value",
        "absolute_difference",
        "source",
        "source_file",
    ]
    assert len(rejected) == 1
    assert rejected.loc[0, "density_g/cm^3"] == 1.25
    assert rejected.loc[0, "retained_value"] == 1.2500005
    assert rejected.loc[0, "absolute_difference"] == pytest.approx(5e-7)
    assert rejected.loc[0, "source"] == "AIonopedia"
    assert rejected.loc[0, "source_file"] == "AIonopedia_density_structured.csv"

    write_csv(ilbert_path, [{**base, "density_g/cm^3": 1.3}])
    merge_data(input_root, output_root)

    assert not (output_root / "rejected_rows").exists()


def test_solvation_revision_pairs_keep_nearest_aionopedia_value(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    base = {
        "cation": "CCCC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "solute": "O=C=O",
        "temperature_K": 298.15,
    }
    outside = {**base, "temperature_K": 308.15}
    write_csv(
        input_root / "AIonopedia" / "AIonopedia_solvation_structured.csv",
        [
            {**base, "solvation_kcal/mol": -0.507},
            {**base, "solvation_kcal/mol": -0.505462},
            {**outside, "solvation_kcal/mol": -1.0},
        ],
    )
    write_csv(
        input_root / "after_AIonopedia" / "after_AIonopedia_solv_structured.csv",
        [
            {**base, "solvation_kcal/mol": -0.507472},
            {**outside, "solvation_kcal/mol": -1.006},
        ],
    )

    merge_data(input_root, output_root)

    solvation = pd.read_csv(output_root / "experiment" / "solvation.csv")
    paired = solvation[solvation["temperature_K"].eq(298.15)]
    assert set(paired["solvation_kcal/mol"]) == {-0.507, -0.505462}
    assert paired.loc[paired["solvation_kcal/mol"].eq(-0.507), "source_list"].item() == (
        "AIonopedia; after_AIonopedia"
    )
    assert set(solvation.loc[solvation["temperature_K"].eq(308.15), "solvation_kcal/mol"]) == {
        -1.0,
        -1.006,
    }

    rejected = pd.read_csv(
        output_root
        / "rejected_rows"
        / "experiment"
        / "solvation_approximate_values_rejected.csv"
    )
    assert len(rejected) == 1
    assert rejected.loc[0, "solvation_kcal/mol"] == -0.507472
    assert rejected.loc[0, "retained_value"] == -0.507
    assert rejected.loc[0, "source"] == "after_AIonopedia"


def test_same_system_condition_rows_are_saved_before_value_collapsing(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    base = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
    }
    paths = {
        "AIonopedia": input_root / "AIonopedia" / "AIonopedia_density_structured.csv",
        "ILBERT": input_root / "ILBERT" / "ILBERT_density_structured.csv",
        "ILThermo": input_root / "ILThermo" / "ilt_density_structured.csv",
        "after_AIonopedia": (
            input_root / "after_AIonopedia" / "after_AIonopedia_density_structured.csv"
        ),
    }
    write_csv(
        paths["AIonopedia"],
        [
            {**base, "density_g/cm^3": 1.2},
            {**base, "temperature_K": 308.15, "density_g/cm^3": 1.4},
        ],
    )
    write_csv(paths["ILBERT"], [{**base, "density_g/cm^3": 1.2000004}])
    write_csv(paths["ILThermo"], [{**base, "density_g/cm^3": 1.2}])
    write_csv(paths["after_AIonopedia"], [{**base, "density_g/cm^3": 1.3}])

    merge_data(input_root, output_root)

    matching_path = output_root / "same_system_condition_rows" / "experiment" / "density.csv"
    matching = pd.read_csv(matching_path)
    assert list(matching.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "pressure_kPa",
        "density_g/cm^3",
        "matching_entry_count",
        "source",
        "source_file",
    ]
    assert len(matching) == 4
    assert matching["temperature_K"].eq(298.15).all()
    assert matching["matching_entry_count"].eq(4).all()
    assert set(matching["density_g/cm^3"]) == {1.2, 1.2000004, 1.3}
    assert set(matching["source"]) == {
        "AIonopedia",
        "ILBERT",
        "ILThermo",
        "after_AIonopedia",
    }

    for source in ("ILBERT", "ILThermo", "after_AIonopedia"):
        paths[source].unlink()
    merge_data(input_root, output_root)

    assert not (output_root / "same_system_condition_rows").exists()


def test_refractive_index_missing_wavelength_defaults_to_sodium_d_line(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    base = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
        "refractive_index_unitless": 1.4,
    }
    write_csv(input_root / "ILBERT" / "ILBERT_refractive_index_structured.csv", [base])
    write_csv(
        input_root / "ILThermo" / "ilt_refractive_index_structured.csv",
        [{**base, "wavelength_nm": 450.0, "refractive_index_unitless": 1.5}],
    )

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "refractive_index.csv")
    assert "wavelength_nm" in out.columns
    assert set(out["wavelength_nm"]) == {450.0, 589.0}
    assert out["wavelength_nm"].isna().sum() == 0


def test_relative_permittivity_labels_merge_into_static_and_dynamic_files(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    base = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
        "pressure_kPa": 101.325,
    }
    write_csv(
        input_root / "ILThermo" / "ilt_static_relative_permittivity_structured.csv",
        [{**base, "static_relative_permittivity_unitless": 12.0}],
    )
    write_csv(
        input_root / "ILThermo" / "ilt_dynamic_relative_permittivity_structured.csv",
        [{**base, "frequency_MHz": 10.0, "dynamic_relative_permittivity_unitless": 8.0}],
    )

    merge_data(input_root, output_root)

    static = pd.read_csv(output_root / "experiment" / "static_relative_permittivity.csv")
    dynamic = pd.read_csv(output_root / "experiment" / "dynamic_relative_permittivity.csv")
    assert "frequency_MHz" not in static.columns
    assert static.loc[0, "static_relative_permittivity_unitless"] == 12.0
    assert dynamic.loc[0, "frequency_MHz"] == 10.0
    assert dynamic.loc[0, "dynamic_relative_permittivity_unitless"] == 8.0


def test_simulation_solvation_merges_as_transfer_organic_without_changing_experiment_properties(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    write_csv(
        input_root / "AIonopedia" / "AIonopedia_solvation_structured.csv",
        [{"cation": "CC[n+]1ccn(C)c1", "anion": "F[B-](F)(F)F", "solvation_kcal/mol": -4.0}],
    )
    write_csv(
        input_root / "AIonopedia" / "AIonopedia_transfer_organic_structured.csv",
        [
            {
                "cation": "CC[n+]1ccn(C)c1",
                "anion": "F[B-](F)(F)F",
                "solute": "CCO",
                "transfer_organic_kcal/mol": 1.5,
            }
        ],
    )
    write_csv(
        input_root / "simulation" / "simulated_combi_qm_solv_structured.csv",
        [{"smiles": "CCO", "solvation_kcal/mol": -2.5}],
    )

    merge_data(input_root, output_root)

    simulation = pd.read_csv(output_root / "simulation" / "transfer_organic.csv")
    assert list(simulation.columns) == ["smiles", "transfer_organic_kcal/mol", "source_list"]
    assert simulation.loc[0, "transfer_organic_kcal/mol"] == -2.5
    assert not (output_root / "simulation" / "solvation.csv").exists()
    assert (output_root / "experiment" / "solvation.csv").exists()
    assert (output_root / "experiment" / "transfer_organic.csv").exists()

    manifest = pd.read_csv(output_root / "merged_manifest.csv")
    simulation_rows = manifest[manifest["bucket"].eq("simulation")]
    assert set(simulation_rows["property_label"]) == {"transfer_organic_kcal/mol"}
    assert simulation_rows.iloc[0]["output_file"] == "simulation/transfer_organic.csv"


def test_simulation_stays_separate_from_experiment_for_same_property(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    row = {"cation": "CC[n+]1ccn(C)c1", "anion": "F[B-](F)(F)F", "temperature_K": 298.15, "density_g/cm^3": 1.2}
    write_csv(input_root / "ILBERT" / "ILBERT_density_structured.csv", [row])
    write_csv(input_root / "simulation" / "simulated_density_structured.csv", [{**row, "density_err_g/cm^3": 0.01}])

    merge_data(input_root, output_root)

    experiment = pd.read_csv(output_root / "experiment" / "density.csv")
    simulation = pd.read_csv(output_root / "simulation" / "density.csv")
    assert len(experiment) == 1
    assert len(simulation) == 1
    assert "density_err_g/cm^3" not in simulation.columns
    assert not (output_root / "simulation" / "density_err.csv").exists()


def test_single_ion_orbitals_are_wide_and_gap_is_audited(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    write_csv(
        input_root / "simulation" / "simulated_HOMO+LUMO_PBE_TZVP_anions_structured.csv",
        [{"anion": "F[B-](F)(F)F", "HOMO_eV": -1.0, "LUMO_eV": 2.0, "gap_eV": 3.0}],
    )
    write_csv(
        input_root / "simulation" / "simulated_HOMO+LUMO_PBE_TZVP_cations_structured.csv",
        [
            {
                "cation": "CC[n+]1ccn(C)c1",
                "HOMO_eV": -9.0,
                "LUMO_eV": -4.0,
                "gap_eV": 4.5,
            }
        ],
    )
    write_csv(
        input_root / "simulation" / "simulated_heat_capacity_structured.csv",
        [
            {
                "cation": "CC[n+]1ccn(C)c1",
                "anion": "F[B-](F)(F)F",
                "temperature_K": 298.15,
                "heat_capacity_J/mol/K": 500.0,
                "heat_capacity_err_J/mol/K": 5.0,
            }
        ],
    )

    merge_data(input_root, output_root)

    simulation_root = output_root / "simulation"
    anion = pd.read_csv(simulation_root / "pbe_tzvp_anion_orbitals.csv")
    cation = pd.read_csv(simulation_root / "pbe_tzvp_cation_orbitals.csv")
    assert list(anion.columns) == ["anion", "HOMO_eV", "LUMO_eV", "source_list"]
    assert list(cation.columns) == ["cation", "HOMO_eV", "LUMO_eV", "source_list"]
    assert anion.loc[0, ["HOMO_eV", "LUMO_eV"]].tolist() == [-1.0, 2.0]
    assert cation.loc[0, ["HOMO_eV", "LUMO_eV"]].tolist() == [-9.0, -4.0]
    assert not (simulation_root / "homo.csv").exists()
    assert not (simulation_root / "lumo.csv").exists()
    assert not (simulation_root / "gap.csv").exists()
    assert not (simulation_root / "anion_homo.csv").exists()
    assert not (simulation_root / "cation_lumo.csv").exists()

    manifest = pd.read_csv(output_root / "merged_manifest.csv")
    orbital_rows = manifest[manifest["property_label"].isin(
        {"pbe_tzvp_anion_orbitals", "pbe_tzvp_cation_orbitals"}
    )]
    assert dict(zip(orbital_rows["property_label"], orbital_rows["output_file"])) == {
        "pbe_tzvp_anion_orbitals": "simulation/pbe_tzvp_anion_orbitals.csv",
        "pbe_tzvp_cation_orbitals": "simulation/pbe_tzvp_cation_orbitals.csv",
    }
    assert set(orbital_rows["input_rows"]) == {1}
    assert set(orbital_rows["output_rows"]) == {1}
    summary = pd.read_csv(output_root / "_audit" / "orbital_gap_consistency_summary.csv")
    assert summary["checked_rows"].tolist() == [1, 1]
    assert summary["exceeded_rows"].tolist() == [0, 1]
    anomalies = pd.read_csv(output_root / "_audit" / "orbital_gap_consistency_anomalies.csv")
    assert len(anomalies) == 1
    assert anomalies.loc[0, "identity_column"] == "cation"
    assert anomalies.loc[0, "absolute_residual_eV"] == pytest.approx(0.5)

    heat_capacity = pd.read_csv(output_root / "simulation" / "heat_capacity.csv")
    assert list(heat_capacity.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "heat_capacity_J/mol/K",
        "source_list",
    ]
    assert not (output_root / "simulation" / "heat_capacity_err.csv").exists()


def test_qm_elec_hf_file_outputs_one_wide_table_instead_of_split_labels(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    write_csv(
        input_root / "simulation" / "simulated_QM_elec_HF_structured.csv",
        [
            {"SMILES": "CCO", "ESP_max": 0.0, "ESP_min": -1.0, "gap_eV": 2.0},
            {"SMILES": "CCO", "ESP_max": 2.0, "ESP_min": -3.0, "gap_eV": 4.0},
            {"SMILES": "CCN", "ESP_max": 0.0, "ESP_min": None, "gap_eV": 1.0},
            {"SMILES": "CCN", "ESP_max": 4.0, "ESP_min": -2.0, "gap_eV": 3.0},
            {"SMILES": "CCN", "ESP_max": 10.0, "ESP_min": -4.0, "gap_eV": 5.0},
            {"SMILES": "CCC", "ESP_max": 7.0, "ESP_min": -7.0, "gap_eV": 6.0},
        ],
    )

    merge_data(input_root, output_root)

    wide = pd.read_csv(output_root / "simulation" / "simulated_qm_elec_hf.csv")
    assert list(wide.columns) == ["SMILES", "ESP_max", "ESP_min", "gap_eV", "source_list"]
    assert len(wide) == 3
    assert wide["SMILES"].is_unique
    cco = wide[wide["SMILES"].eq("CCO")].iloc[0]
    assert cco[["ESP_max", "ESP_min", "gap_eV"]].tolist() == [1.0, -2.0, 3.0]
    ccn = wide[wide["SMILES"].eq("CCN")].iloc[0]
    assert ccn[["ESP_max", "ESP_min", "gap_eV"]].tolist() == [4.0, -3.0, 3.0]
    assert set(wide["source_list"]) == {"simulation"}
    assert not (output_root / "simulation" / "esp_max.csv").exists()
    assert not (output_root / "simulation" / "esp_min.csv").exists()
    assert not (output_root / "simulation" / "gap.csv").exists()

    manifest = pd.read_csv(output_root / "merged_manifest.csv")
    assert manifest.iloc[0]["property_label"] == "simulated_QM_elec_HF"
    assert manifest.iloc[0]["output_file"] == "simulation/simulated_qm_elec_hf.csv"


def test_3d_box_file_outputs_one_wide_table(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    row = {
        "mol_id": "mol_1",
        "cation": "C[N+](C)(C)C",
        "anion": "[Cl-]",
        "temperature_K": 298.15,
        "rdf_ca_peak1_r_A": 4.1,
        "rdf_ca_peak1_g": 2.1,
        "rdf_ca_coordination1": 5.1,
        "rdf_ca_excess_area_A": 0.8,
        "rdf_cc_peak1_g": 1.5,
        "rdf_cc_excess_area_A": 0.3,
        "rdf_aa_peak1_g": 1.7,
        "rdf_aa_excess_area_A": 0.5,
        "scc_prepeak_present": True,
        "scc_prepeak_q_A^-1": 0.25,
        "scc_prepeak_height": 1.4,
        "scc_prepeak_area_A^-1": 0.12,
        "szz_peak_q_A^-1": 0.7,
        "szz_peak_height": 1.9,
        "szz_peak_area_A^-1": 0.2,
    }
    write_csv(input_root / "simulation" / "3d_box_structured.csv", [row])

    merge_data(input_root, output_root)

    wide = pd.read_csv(output_root / "simulation" / "3d_box.csv")
    assert list(wide.columns) == [*row, "source_list"]
    assert wide.loc[0, "source_list"] == "simulation"
    assert not (output_root / "simulation" / "rdf_ca_peak1_r_a.csv").exists()
    manifest = pd.read_csv(output_root / "merged_manifest.csv")
    assert manifest.iloc[0]["property_label"] == "3d_box"
    assert manifest.iloc[0]["output_file"] == "simulation/3d_box.csv"


def test_same_key_different_label_values_remain_separate_rows(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    base = {"cation": "CC[n+]1ccn(C)c1", "anion": "F[B-](F)(F)F", "temperature_K": 298.15}
    write_csv(input_root / "AIonopedia" / "AIonopedia_viscosity_structured.csv", [{**base, "viscosity_mPa*s_log10": 1.0}])
    write_csv(input_root / "ILBERT" / "ILBERT_viscosity_structured.csv", [{**base, "viscosity_mPa*s_log10": 2.0}])

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "viscosity.csv")
    assert len(out) == 2
    assert set(out["viscosity_mPa*s_log10"]) == {1.0, 2.0}
    assert set(out["source_list"]) == {"AIonopedia", "ILBERT"}


def test_condition_subset_row_collapses_into_more_complete_record(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    partial = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
        "surface_tension_mN/m": 30.0,
    }
    complete = {**partial, "pressure_kPa": 100.0}
    write_csv(input_root / "AIonopedia" / "AIonopedia_surface_tension_structured.csv", [partial])
    write_csv(input_root / "ILBERT" / "ILBERT_surface_tension_structured.csv", [complete])

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "surface_tension.csv")
    assert len(out) == 1
    assert out.iloc[0]["pressure_kPa"] == 100.0
    assert out.iloc[0]["source_list"] == "AIonopedia; ILBERT"


def test_condition_subset_does_not_collapse_different_label_values(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    base = {"cation": "CC[n+]1ccn(C)c1", "anion": "F[B-](F)(F)F", "temperature_K": 298.15}
    write_csv(input_root / "AIonopedia" / "AIonopedia_density_structured.csv", [{**base, "density_g/cm^3": 1.2}])
    write_csv(
        input_root / "ILBERT" / "ILBERT_density_structured.csv",
        [{**base, "pressure_kPa": 100.0, "density_g/cm^3": 1.3}],
    )

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "density.csv")
    assert len(out) == 2
    assert set(out["density_g/cm^3"]) == {1.2, 1.3}


def test_ambiguous_condition_subset_candidates_are_preserved(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    partial = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
        "density_g/cm^3": 1.2,
    }
    write_csv(input_root / "AIonopedia" / "AIonopedia_density_structured.csv", [partial])
    write_csv(
        input_root / "ILBERT" / "ILBERT_density_structured.csv",
        [
            {**partial, "pressure_kPa": 100.0},
            {**partial, "pressure_kPa": 200.0},
        ],
    )

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "density.csv")
    assert len(out) == 3
    assert set(out["pressure_kPa"]) == {100.0, 101.325, 200.0}


def test_complementary_condition_rows_are_preserved(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    base = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
        "density_g/cm^3": 1.2,
    }
    write_csv(input_root / "AIonopedia" / "AIonopedia_density_structured.csv", [{**base, "phase": "Liquid"}])
    write_csv(input_root / "ILBERT" / "ILBERT_density_structured.csv", [{**base, "pressure_kPa": 100.0}])

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "density.csv")
    assert len(out) == 2
    assert set(out["source_list"]) == {"AIonopedia", "ILBERT"}


def test_standard_state_note_is_metadata_not_property(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    reference_note = (
        "* property value is given as the difference ( x - x ref ) from the reference state: "
        "Crystal at the temperature of 298.15 K and the pressure of 101.3 kPa"
    )
    write_csv(
        input_root / "ILThermo" / "ilt_enthalpy_structured.csv",
        [
            {
                "cation": "CCCC[n+]1ccccc1",
                "anion": "F[B-](F)(F)F",
                "temperature_K": 290.0,
                "pressure_kPa": 101.325,
                "standard_state_note": reference_note,
                "enthalpy_kJ/mol": -4.329,
            }
        ],
    )

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "enthalpy.csv")
    assert list(out.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "pressure_kPa",
        "standard_state_note",
        "enthalpy_kJ/mol",
        "source_list",
    ]
    assert out.loc[0, "standard_state_note"] == reference_note
    assert out.loc[0, "enthalpy_kJ/mol"] == -4.329
    assert not (output_root / "experiment" / "standard_state_note.csv").exists()


def test_ilthermo_equilibrium_pressure_uses_specific_output_filename(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    write_csv(
        input_root / "ILThermo" / "ilt_equilibrium_pressure_structured.csv",
        [
            {
                "cation": "CC[n+]1ccn(C)c1",
                "anion": "F[B-](F)(F)F",
                "temperature_K": 298.15,
                "pressure_kPa_log10": 2.0,
            }
        ],
    )

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "equilibrium_pressure.csv")
    assert list(out.columns) == ["cation", "anion", "temperature_K", "pressure_kPa_log10", "source_list"]
    assert out.loc[0, "pressure_kPa_log10"] == 2.0
    assert not (output_root / "experiment" / "pressure.csv").exists()
    manifest = pd.read_csv(output_root / "merged_manifest.csv")
    assert manifest.iloc[0]["property_label"] == "pressure_kPa_log10"
    assert manifest.iloc[0]["output_file"] == "experiment/equilibrium_pressure.csv"


def test_self_diffusion_log_label_keeps_stable_output_filename(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    write_csv(
        input_root / "ILThermo" / "ilt_self_diffusion_coefficient_structured.csv",
        [
            {
                "cation": "CC[n+]1ccn(C)c1",
                "anion": "F[B-](F)(F)F",
                "temperature_K": 298.15,
                "self_diffusion_coefficient_10^-9*m^2/s_log10": -2.0,
            }
        ],
    )

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "self_diffusion_coefficient.csv")
    assert list(out.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "self_diffusion_coefficient_10^-9*m^2/s_log10",
        "source_list",
    ]
    assert out.loc[0, "self_diffusion_coefficient_10^-9*m^2/s_log10"] == -2.0
    manifest = pd.read_csv(output_root / "merged_manifest.csv")
    assert manifest.iloc[0]["property_label"] == "self_diffusion_coefficient_10^-9*m^2/s_log10"
    assert manifest.iloc[0]["output_file"] == "experiment/self_diffusion_coefficient.csv"


def test_manifest_records_output_files_and_counts(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    stale_system_counts = output_root / "system_property_counts.csv"
    stale_system_counts.parent.mkdir(parents=True, exist_ok=True)
    stale_system_counts.write_text("stale\n")
    write_csv(
        input_root / "ILBERT" / "ILBERT_HC_structured.csv",
        [{"cation": "CC[n+]1ccn(C)c1", "anion": "F[B-](F)(F)F", "temperature_K": 298.15, "heat_capacity_J/mol/K": 500.0}],
    )

    merge_data(input_root, output_root)

    manifest = pd.read_csv(output_root / "merged_manifest.csv")
    assert list(manifest.columns) == ["bucket", "property_label", "output_file", "input_files", "input_rows", "output_rows"]
    row = manifest.iloc[0]
    assert row["bucket"] == "experiment"
    assert row["property_label"] == "heat_capacity_J/mol/K"
    assert row["output_file"] == "experiment/heat_capacity.csv"
    assert row["input_files"] == "ILBERT/ILBERT_HC_structured.csv"
    assert row["input_rows"] == 1
    assert row["output_rows"] == 1
    assert not stale_system_counts.exists()


def test_property_filename_collisions_raise_clear_error(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    row = {"cation": "CC[n+]1ccn(C)c1", "anion": "F[B-](F)(F)F"}
    write_csv(input_root / "ILBERT" / "a.csv", [{**row, "temperature_K": 298.15, "density_g/cm^3": 1.2}])
    write_csv(input_root / "ILBERT" / "b.csv", [{**row, "temperature_K": 298.15, "density_kg/m^3": 1200.0}])

    with pytest.raises(ValueError, match="experiment/density.csv"):
        merge_data(input_root, output_root)
