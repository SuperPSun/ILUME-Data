import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_training_splits import (
    PubChemClient,
    TaskSpec,
    TrainingSplitError,
    augment_pretraining_entities,
    build_training_splits,
    candidate_allowed,
    canonicalize_smiles,
    entity_id,
    extract_pretraining_entities,
    generate_rule_candidates,
    group_id,
    prepare_task_frame,
    role_fragments,
    stable_fraction,
    tier_for_system_count,
)


QM_COLUMNS = [
    "ESP_max",
    "ESP_min",
    "ESP_std",
    "ESP_pos_frac",
    "Dipole",
    "Quadrupole",
    "q_max",
    "q_min",
    "q_std",
    "q_pos_frac",
    "gap_eV",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def ion_pair(index: int) -> tuple[str, str]:
    return f"[{index}Na+]", f"[{index}Cl-]"


def neutral_carbon(index: int) -> str:
    return f"[{index}CH4]"


def neutral_oxygen(index: int) -> str:
    return f"[{index}OH2]"


def write_stage2_files(final_root: Path, count: int = 100) -> None:
    pairs = [ion_pair(index) for index in range(1, count + 1)]
    for filename, target in (
        ("density.csv", "density_g/cm^3"),
        ("heat_capacity.csv", "heat_capacity_J/mol/K"),
        ("thermal_expansion.csv", "thermal_expansion_K^-1"),
    ):
        write_csv(
            final_root / "simulation" / filename,
            [
                {
                    "cation": cation,
                    "anion": anion,
                    "temperature_K": 298.15,
                    target: float(index),
                    "source_list": "simulation",
                }
                for index, (cation, anion) in enumerate(pairs)
            ],
        )
    write_csv(
        final_root / "simulation" / "simulated_qm_elec_hf.csv",
        [
            {
                "SMILES": neutral_carbon(index),
                **{column: float(index) for column in QM_COLUMNS},
                "source_list": "simulation",
            }
            for index in range(1, count + 1)
        ],
    )
    write_csv(
        final_root / "simulation" / "transfer_organic.csv",
        [
            {
                "solute": neutral_carbon(index),
                "solvent": neutral_oxygen(index),
                "transfer_organic_kcal/mol": float(index),
                "source_list": "simulation",
            }
            for index in range(1, count + 1)
        ],
    )


def test_extract_pretraining_entities_splits_ion_fragments_and_preserves_roles(
    tmp_path: Path,
):
    final_root = tmp_path / "final"
    write_csv(
        final_root / "experiment" / "solvation.csv",
        [
            {
                "cation": "[Na+].[K+]",
                "anion": "[Cl-].[Br-]",
                "solute": "CCO",
                "temperature_K": 298.15,
                "solvation_kcal/mol": -2.0,
                "source_list": "test",
            }
        ],
    )
    write_csv(
        final_root / "simulation" / "molecules.csv",
        [
            {
                "SMILES": "OCC",
                "value": 1.0,
                "source_list": "test",
            }
        ],
    )
    structure_path = final_root / "simulation" / "charge_20260514" / "bad.mol"
    structure_path.parent.mkdir(parents=True)
    structure_path.write_text("not a molecule", encoding="utf-8")

    output_root = tmp_path / "training"
    entities = extract_pretraining_entities(
        final_root,
        output_root,
        pretrain_cap=100,
    )

    assert set(entities["role"]) == {
        "cation",
        "anion",
        "solute",
        "molecule",
    }
    assert set(entities.loc[entities["role"].eq("cation"), "SMILES"]) == {
        "[K+]",
        "[Na+]",
    }
    assert set(entities.loc[entities["role"].eq("anion"), "SMILES"]) == {
        "[Br-]",
        "[Cl-]",
    }
    ethanol = entities[entities["SMILES"].eq("CCO")]
    assert set(ethanol["role"]) == {"solute", "molecule"}
    assert "il" not in set(entities["role"])

    sources = pd.read_csv(output_root / "stage1" / "entity_sources.csv")
    assert len(sources) == 6
    assert set(sources["source_file"]) == {
        "experiment/solvation.csv",
        "simulation/molecules.csv",
    }
    assert structure_path.exists()


def test_rule_candidates_are_finite_and_preserve_charge():
    butane = {
        (row["rule"], row["SMILES"])
        for row in generate_rule_candidates("CCCC")
    }
    assert ("terminal_alkyl_shorten_1", "CCC") in butane
    assert ("terminal_alkyl_shorten_2", "CC") in butane
    assert ("terminal_alkyl_extend_1", "CCCCC") in butane
    assert ("terminal_alkyl_extend_2", "CCCCCC") in butane

    fluoroethane = {
        (row["rule"], row["SMILES"])
        for row in generate_rule_candidates("CCF")
    }
    assert ("halogen_F_to_Cl", "CCCl") in fluoroethane
    assert ("halogen_F_to_Br", "CCBr") in fluoroethane

    charged = generate_rule_candidates("C[NH3+]")
    assert charged
    assert all(candidate_allowed("C[NH3+]", row["SMILES"], "cation") for row in charged)
    assert not candidate_allowed("C[NH3+]", "CCN", "cation")
    assert not candidate_allowed("[Cl-]", "[Na+]", "anion")


def test_mixed_internal_ion_fragments_remain_one_entity():
    ferrocenyl_anion = (
        "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)c1cc[cH-]c1."
        "[Fe+2].c1cc[cH-]c1"
    )
    fragments = role_fragments(ferrocenyl_anion, "anion")
    assert len(fragments) == 1
    assert fragments[0] == canonicalize_smiles(ferrocenyl_anion)


def test_pubchem_client_retries_caches_and_supports_offline_mode(tmp_path: Path):
    calls: list[str] = []
    sleeps: list[float] = []
    payload = json.dumps(
        {
            "PropertyTable": {
                "Properties": [
                    {"CID": 702, "SMILES": "CCO", "Charge": 0},
                ]
            }
        }
    ).encode()

    def transport(http_request):
        calls.append(http_request.full_url)
        if len(calls) == 1:
            return 503, b"", {}
        return 200, payload, {}

    cache_path = tmp_path / "pubchem.sqlite"
    client = PubChemClient(
        cache_path,
        rate_limit=0,
        max_retries=2,
        transport=transport,
        sleep=sleeps.append,
    )
    records = client.similarity("CCO", threshold=90, max_records=10)
    assert records == [{"CID": 702, "SMILES": "CCO", "Charge": 0}]
    assert len(calls) == 2
    assert sleeps == [1.0]
    assert client.similarity("OCC", threshold=90, max_records=10) == records
    assert len(calls) == 2
    client.close()

    def forbidden_transport(_):
        raise AssertionError("offline cache should not access the network")

    offline = PubChemClient(
        cache_path,
        offline=True,
        rate_limit=0,
        transport=forbidden_transport,
    )
    assert offline.similarity("CCO", threshold=90, max_records=10) == records
    offline.close()


class FakePubChemClient:
    def similarity(
        self,
        smiles: str,
        *,
        threshold: int,
        max_records: int,
    ) -> list[dict[str, object]]:
        assert threshold == 90
        assert max_records == 100
        assert canonicalize_smiles(smiles) == "CCF"
        return [
            {"CID": 1, "SMILES": "CCF", "Charge": 0},
            {"CID": 2, "SMILES": "CCBr", "Charge": 0},
            {"CID": 3, "SMILES": "C[NH3+]", "Charge": 1},
        ]

    def identity(self, smiles: str) -> list[dict[str, object]]:
        if canonicalize_smiles(smiles) == "CCCl":
            return [{"CID": 4, "SMILES": "CCCl", "Charge": 0}]
        return []


def test_augmentation_uses_global_cap_and_records_selected_provenance(
    tmp_path: Path,
):
    output_root = tmp_path / "training"
    stage1 = output_root / "stage1"
    stage1.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "entity_id": entity_id("molecule", "CCF"),
                "role": "molecule",
                "SMILES": "CCF",
                "formal_charge": 0,
                "is_original": True,
            }
        ]
    ).to_csv(stage1 / "entities.csv", index=False)
    pd.DataFrame(
        [
            {
                "entity_id": entity_id("molecule", "CCF"),
                "source_file": "simulation/example.csv",
                "source_column": "SMILES",
                "source_row": 0,
                "fragment_index": 0,
            }
        ]
    ).to_csv(stage1 / "entity_sources.csv", index=False)

    augmented = augment_pretraining_entities(
        output_root,
        pretrain_cap=3,
        client=FakePubChemClient(),
    )

    assert len(augmented) == 3
    assert set(augmented["SMILES"]) == {"CCF", "CCCl", "CCBr"}
    assert int(augmented["is_original"].sum()) == 1
    provenance = pd.read_csv(stage1 / "augmentation_provenance.csv")
    selected = provenance[provenance["status"].eq("selected")]
    assert set(selected["method"]) == {"rule", "pubchem_similarity"}
    assert set(selected["SMILES"]) == {"CCCl", "CCBr"}


@pytest.mark.parametrize(
    ("systems", "expected"),
    [
        (49, "small"),
        (50, "medium"),
        (500, "medium"),
        (501, "large"),
    ],
)
def test_tier_boundaries(systems: int, expected: str):
    assert tier_for_system_count(systems) == expected


def test_charge_rows_are_deduplicated_and_conflicts_fail(tmp_path: Path):
    final_root = tmp_path / "final"
    path = final_root / "simulation" / "charge.csv"
    write_csv(
        path,
        [
            {
                "mol_id": "mol_1",
                "SMILES": "OCC",
                "charge": 0,
                "source_list": "simulation",
            },
            {
                "mol_id": "mol_2",
                "SMILES": "CCO",
                "charge": 0,
                "source_list": "simulation",
            },
        ],
    )
    task = TaskSpec(
        task_id="simulation/charge",
        stage=3,
        source_file="simulation/charge.csv",
        target_columns=("charge",),
        identity_columns=("SMILES",),
        system_type="molecule",
    )

    frame, raw_rows = prepare_task_frame(final_root, task, "checksum")
    assert raw_rows == 2
    assert len(frame) == 1
    assert frame.iloc[0]["SMILES"] == "CCO"
    assert frame.iloc[0]["_mol_ids"] == "mol_1;mol_2"

    conflicting = pd.read_csv(path)
    conflicting.loc[1, "charge"] = -1
    conflicting.to_csv(path, index=False)
    with pytest.raises(TrainingSplitError, match="Conflicting charge labels"):
        prepare_task_frame(final_root, task, "checksum")


def test_build_training_splits_end_to_end_is_disjoint_and_deterministic(
    tmp_path: Path,
):
    final_root = tmp_path / "final"
    write_stage2_files(final_root)

    large_pairs = [ion_pair(index) for index in range(1, 502)]
    for filename, target in (
        ("density.csv", "density_g/cm^3"),
        ("electrical_conductivity.csv", "electrical_conductivity_S/m_log10"),
    ):
        write_csv(
            final_root / "experiment" / filename,
            [
                {
                    "cation": cation,
                    "anion": anion,
                    "temperature_K": 298.15,
                    "pressure_kPa": 101.325,
                    target: float(index),
                    "source_list": "experiment",
                }
                for index, (cation, anion) in enumerate(large_pairs)
            ],
        )

    selected_indices: list[int] = []
    unselected_indices: list[int] = []
    for index, (cation, anion) in enumerate(large_pairs, start=1):
        system_identifier = group_id("il", (cation, anion))
        destination = (
            selected_indices
            if stable_fraction(42, "stage3_test", "il", system_identifier) < 0.1
            else unselected_indices
        )
        destination.append(index)
    small_indices = selected_indices[:10] + unselected_indices[:39]
    assert len(small_indices) == 49
    write_csv(
        final_root / "experiment" / "dynamic_relative_permittivity.csv",
        [
            {
                "cation": ion_pair(index)[0],
                "anion": ion_pair(index)[1],
                "temperature_K": 298.15,
                "pressure_kPa": 101.325,
                "frequency_MHz": 1.0,
                "dynamic_relative_permittivity_unitless": float(index),
                "source_list": "experiment",
            }
            for index in small_indices
        ],
    )
    write_csv(
        final_root / "experiment" / "solvation.csv",
        [
            {
                "cation": ion_pair(index)[0],
                "anion": ion_pair(index)[1],
                "solute": neutral_carbon(index),
                "temperature_K": 298.15,
                "solvation_kcal/mol": float(index),
                "source_list": "experiment",
            }
            for index in range(1, 51)
        ],
    )
    write_csv(
        final_root / "experiment" / "transfer_organic.csv",
        [
            {
                "solute": neutral_carbon(index),
                "solvent": neutral_oxygen(index),
                "temperature_K": 298.15,
                "transfer_organic_kcal/mol": float(index),
                "source_list": "experiment",
            }
            for index in range(1, 51)
        ],
    )
    ignored_structure = (
        final_root / "simulation" / "charge_20260514" / "invalid.mol2"
    )
    ignored_structure.parent.mkdir(parents=True)
    ignored_structure.write_text("invalid structure", encoding="utf-8")

    output_root = tmp_path / "training"
    extract_pretraining_entities(
        final_root,
        output_root,
        pretrain_cap=10_000,
    )
    catalog = build_training_splits(final_root, output_root, seed=42)

    assert int(catalog["stage"].eq(2).sum()) == 5
    assert set(
        catalog.loc[catalog["stage"].eq(2), "task_id"]
    ) == {
        "simulation/density",
        "simulation/heat_capacity",
        "simulation/simulated_qm_elec_hf",
        "simulation/thermal_expansion",
        "simulation/transfer_organic",
    }
    stage3 = catalog[catalog["stage"].eq(3)].set_index("task_id")
    assert stage3.loc["experiment/density", "tier"] == "large"
    assert stage3.loc["experiment/solvation", "tier"] == "medium"
    assert (
        stage3.loc["experiment/dynamic_relative_permittivity", "tier"]
        == "small"
    )
    assert stage3.loc["experiment/density", "repeats"] == 1
    assert stage3.loc["experiment/solvation", "repeats"] == 5

    test_groups = pd.read_csv(output_root / "stage3" / "test_groups.csv")
    shared_large = test_groups[
        test_groups["source_tasks"].str.contains("experiment/density")
        & test_groups["source_tasks"].str.contains(
            "experiment/electrical_conductivity"
        )
    ]
    assert not shared_large.empty

    stage3_rows = pd.read_csv(output_root / "stage3" / "rows.csv")
    small_rows = stage3_rows[
        stage3_rows["task_id"].eq(
            "experiment/dynamic_relative_permittivity"
        )
    ]
    assert (
        small_rows["partition"].eq("reserved_due_to_cross_task_test").sum()
        == 10
    )
    random_assignments = pd.read_csv(
        output_root / "stage3" / "random_fold_assignments.csv"
    )
    assert set(
        small_rows.loc[
            small_rows["partition"].eq("reserved_due_to_cross_task_test"),
            "row_id",
        ]
    ).isdisjoint(set(random_assignments["row_id"]))

    medium_random = random_assignments[
        random_assignments["task_id"].eq("experiment/solvation")
    ]
    assert set(medium_random["repeat"]) == set(range(5))
    large_random = random_assignments[
        random_assignments["task_id"].eq("experiment/density")
    ]
    assert set(large_random["repeat"]) == {0}

    group_folds = pd.read_csv(
        output_root / "stage3" / "group_fold_assignments.csv"
    )
    assert not group_folds.duplicated(
        ["strategy", "group_id", "repeat"]
    ).any()
    loo = pd.read_csv(output_root / "stage3" / "loo_groups.csv")
    assert set(loo["task_id"]) == {
        "experiment/dynamic_relative_permittivity"
    }
    overlap_checks = pd.read_csv(
        output_root / "audit" / "overlap_checks.csv"
    )
    assert overlap_checks["passed"].all()
    assert ignored_structure.exists()

    checksummed_paths = [
        output_root / "task_catalog.csv",
        output_root / "stage2" / "group_assignments.csv",
        output_root / "stage3" / "group_fold_assignments.csv",
        output_root / "audit" / "label_and_fold_summary.csv",
    ]
    first_run = {path: path.read_bytes() for path in checksummed_paths}
    build_training_splits(final_root, output_root, seed=42)
    assert first_run == {path: path.read_bytes() for path in checksummed_paths}
