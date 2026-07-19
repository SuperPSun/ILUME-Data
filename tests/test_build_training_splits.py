import json
from http.client import IncompleteRead
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

from scripts.build_training_splits import (
    PRETRAIN_ENTITY_COLUMNS,
    PubChemClient,
    TaskProfile,
    TaskSpec,
    TrainingSplitError,
    _round_robin_rule_candidates,
    augment_pretraining_entities,
    build_training_splits,
    candidate_allowed,
    canonicalize_smiles,
    extract_pretraining_entities,
    fixed_h_identity_key,
    formal_charge,
    generate_rule_candidates,
    plan_joint_test_registry,
    plan_shared_group_folds,
    plan_shared_weighted_group_folds,
    prepare_task_frame,
    role_fragments,
    tier_for_system_count,
    validate_final_identity_consistency,
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
    cation_chain = (index - 1) % 23 + 1
    anion_chain = ((index - 1) * 7) % 25 + 1
    return (
        f"{'C' * cation_chain}[N+](C)(C)C",
        f"{'C' * anion_chain}(=O)[O-]",
    )


def neutral_carbon(index: int) -> str:
    return "C" * index


def neutral_oxygen(index: int) -> str:
    return "O" + ("C" * index)


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

    assert {
        path.name
        for path in (output_root / "stage1").glob("*.csv")
    } == {
        "IL.csv",
        "anion.csv",
        "cation.csv",
        "solute.csv",
        "solvent.csv",
        "molecule.csv",
    }
    il = pd.read_csv(output_root / "stage1" / "IL.csv")
    assert list(il.columns) == ["cation", "anion"]
    assert il.to_dict(orient="records") == [
        {
            "cation": canonicalize_smiles("[Na+].[K+]"),
            "anion": canonicalize_smiles("[Cl-].[Br-]"),
        }
    ]
    molecule = pd.read_csv(output_root / "stage1" / "molecule.csv")
    assert list(molecule.columns) == PRETRAIN_ENTITY_COLUMNS
    assert molecule.loc[0, "origin_list"] == "dataset"
    assert not (output_root / "stage1" / "entities.csv").exists()
    assert not (output_root / "stage1" / "entity_sources.csv").exists()
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


def test_rule_candidates_include_all_valid_resonance_forms():
    seed = canonicalize_smiles("CCCC[n+]1ccn(C)c1")
    molecule = Chem.MolFromSmiles(seed)
    assert molecule is not None
    expected = {
        canonicalize_smiles(
            Chem.MolToSmiles(
                resonance,
                canonical=True,
                isomericSmiles=True,
            )
        )
        for resonance in Chem.ResonanceMolSupplier(
            molecule,
            0,
            100,
        )
    }
    expected.discard(seed)
    expected = {
        smiles
        for smiles in expected
        if fixed_h_identity_key(smiles) == fixed_h_identity_key(seed)
    }

    generated = generate_rule_candidates(
        seed,
        "cation",
        resonance_max_structs=100,
    )
    resonance = {
        row["SMILES"]
        for row in generated
        if row["rule"] == "resonance_equivalent"
    }

    assert resonance == expected
    assert resonance
    assert all(
        fixed_h_identity_key(smiles) == fixed_h_identity_key(seed)
        for smiles in resonance
    )
    assert all(
        formal_charge(smiles) == formal_charge(seed)
        for smiles in resonance
    )


def test_final_identity_validation_rejects_equivalent_smiles_variants(
    tmp_path: Path,
):
    final_root = tmp_path / "final"
    write_csv(
        final_root / "experiment" / "density.csv",
        [
            {
                "cation": "CCCC[n+]1ccn(C)c1",
                "anion": "[Cl-]",
                "density_g/cm^3": 1.0,
            }
        ],
    )
    write_csv(
        final_root / "simulation" / "density.csv",
        [
            {
                "cation": "CCCCn1cc[n+](C)c1",
                "anion": "[Cl-]",
                "density_g/cm^3": 1.0,
            }
        ],
    )

    with pytest.raises(
        TrainingSplitError,
        match="Rebuild data/merged",
    ):
        validate_final_identity_consistency(final_root)


def test_ion_rules_extend_and_branch_terminal_alkyl_chains():
    linear = {
        (row["rule"], row["SMILES"])
        for row in generate_rule_candidates(
            "CCC[N+](C)(C)C",
            "cation",
        )
    }
    assert (
        "terminal_alkyl_extend_4",
        canonicalize_smiles("CCCCCCC[N+](C)(C)C"),
    ) in linear
    assert (
        "terminal_alkyl_linear_to_branch",
        canonicalize_smiles("CC(C)[N+](C)(C)C"),
    ) in linear

    branched = {
        (row["rule"], row["SMILES"])
        for row in generate_rule_candidates(
            "CC(C)[N+](C)(C)C",
            "cation",
        )
    }
    assert (
        "terminal_alkyl_branch_to_linear",
        canonicalize_smiles("CCC[N+](C)(C)C"),
    ) in branched

    neutral_rules = {
        row["rule"]
        for row in generate_rule_candidates("CCCC", "molecule")
    }
    assert "terminal_alkyl_extend_3" not in neutral_rules
    assert "terminal_alkyl_linear_to_branch" not in neutral_rules


def test_rule_candidates_drop_disconnected_explicit_hydrogen_fragments(capfd):
    seed = "[2H]C([2H])([2H])C([2H])([2H])[n+]1ccn(C)c1"
    candidates = generate_rule_candidates(seed, "cation")

    assert candidates
    assert all(
        candidate_allowed(seed, row["SMILES"], "cation")
        for row in candidates
    )
    assert "not removing hydrogen atom without neighbors" not in capfd.readouterr().err


def test_ion_rules_cover_headgroups_and_perfluoroalkyl_chains():
    ammonium = {
        (row["rule"], row["SMILES"])
        for row in generate_rule_candidates("C[N+](C)(C)C", "cation")
    }
    assert (
        "cation_headgroup_N_to_P",
        canonicalize_smiles("C[P+](C)(C)C"),
    ) in ammonium

    phosphonium = {
        (row["rule"], row["SMILES"])
        for row in generate_rule_candidates("C[P+](C)(C)C", "cation")
    }
    assert (
        "cation_headgroup_P_to_N",
        canonicalize_smiles("C[N+](C)(C)C"),
    ) in phosphonium
    assert not any(
        row["rule"].startswith("cation_headgroup")
        for row in generate_rule_candidates("C[N+](C)(C)C", "molecule")
    )

    triflate = {
        (row["rule"], row["SMILES"])
        for row in generate_rule_candidates(
            "[O-]S(=O)(=O)C(F)(F)F",
            "anion",
        )
    }
    pentafluoroethyl = canonicalize_smiles(
        "[O-]S(=O)(=O)C(F)(F)C(F)(F)F"
    )
    assert ("perfluoroalkyl_extend_1", pentafluoroethyl) in triflate

    shortened = {
        (row["rule"], row["SMILES"])
        for row in generate_rule_candidates(pentafluoroethyl, "anion")
    }
    assert (
        "perfluoroalkyl_shorten_1",
        canonicalize_smiles("[O-]S(=O)(=O)C(F)(F)F"),
    ) in shortened


def test_ion_rules_cover_chalcogen_aromatic_and_iodine_analogs():
    hydroxyl = {
        (row["rule"], row["SMILES"])
        for row in generate_rule_candidates(
            "OCC[N+](C)(C)C",
            "cation",
        )
    }
    assert (
        "hydroxyl_thiol_O_to_S",
        canonicalize_smiles("SCC[N+](C)(C)C"),
    ) in hydroxyl

    formate = {
        (row["rule"], row["SMILES"])
        for row in generate_rule_candidates("[O-]C=O", "anion")
    }
    assert (
        "anionic_chalcogen_O_to_S",
        canonicalize_smiles("[S-]C=O"),
    ) in formate
    assert (
        "carbonyl_thiocarbonyl_O_to_S",
        canonicalize_smiles("[O-]C=S"),
    ) in formate

    pyridinium = generate_rule_candidates("C[n+]1ccccc1", "cation")
    assert any(row["rule"] == "aromatic_C_to_N" for row in pyridinium)
    assert all(
        candidate_allowed("C[n+]1ccccc1", row["SMILES"], "cation")
        for row in pyridinium
    )

    chloride = {
        (row["rule"], row["SMILES"])
        for row in generate_rule_candidates("[Cl-]", "anion")
    }
    assert ("halogen_Cl_to_I", "[I-]") in chloride
    assert not any(
        "_to_I" in row["rule"] or "I_to_" in row["rule"]
        for row in generate_rule_candidates("CCCl", "molecule")
    )


def test_rule_family_order_is_deterministic_and_round_robin():
    candidates = [
        {
            "candidate_smiles": f"C{index}",
            "method": "rule",
            "pubchem_cid": "",
            "rule": f"{family}_{index}",
            "family": family,
        }
        for family in ("halogen", "terminal_alkyl", "aromatic_CH_N")
        for index in range(3)
    ]
    first = _round_robin_rule_candidates(candidates, "seed")
    second = _round_robin_rule_candidates(candidates, "seed")

    assert first == second
    assert {row["family"] for row in first[:3]} == {
        "halogen",
        "terminal_alkyl",
        "aromatic_CH_N",
    }


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


def test_pubchem_client_retries_incomplete_chunked_response(tmp_path: Path):
    calls = 0
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

    def transport(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise IncompleteRead(b'{"PropertyTable":')
        return 200, payload, {}

    cache_path = tmp_path / "pubchem.sqlite"
    client = PubChemClient(
        cache_path,
        rate_limit=0,
        max_retries=2,
        transport=transport,
        sleep=sleeps.append,
    )

    assert client.similarity("CCO") == [
        {"CID": 702, "SMILES": "CCO", "Charge": 0}
    ]
    assert calls == 2
    assert sleeps == [1.0]
    assert client.similarity("CCO") == [
        {"CID": 702, "SMILES": "CCO", "Charge": 0}
    ]
    assert calls == 2
    client.close()


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
            {"CID": 3, "SMILES": "CCO", "Charge": 0},
        ]

    def identity(self, _smiles: str) -> list[dict[str, object]]:
        raise AssertionError("local rules must not call PubChem identity")


class EmptyPubChemClient:
    def similarity(
        self,
        _smiles: str,
        *,
        threshold: int,
        max_records: int,
    ) -> list[dict[str, object]]:
        assert threshold == 90
        assert max_records == 100
        return []

    def identity(self, _smiles: str) -> list[dict[str, object]]:
        raise AssertionError("resonance rules must not call PubChem identity")


def test_augmentation_keeps_resonance_forms_as_independent_rows(
    tmp_path: Path,
):
    output_root = tmp_path / "training"
    stage1 = output_root / "stage1"
    stage1.mkdir(parents=True)
    pd.DataFrame(
        [{"cation": "CCCC[n+]1ccn(C)c1", "anion": "[Cl-]"}]
    ).to_csv(stage1 / "IL.csv", index=False)
    for role in ("anion", "solute", "solvent", "molecule"):
        pd.DataFrame(columns=PRETRAIN_ENTITY_COLUMNS).to_csv(
            stage1 / f"{role}.csv",
            index=False,
        )
    pd.DataFrame(
        [
            {
                "SMILES": "CCCC[n+]1ccn(C)c1",
                "formal_charge": 1,
                "origin_list": "dataset",
                "seed_smiles_list": "",
                "rule_list": "",
                "pubchem_cid_list": "",
                "mol_id_list": "",
            }
        ],
        columns=PRETRAIN_ENTITY_COLUMNS,
    ).to_csv(stage1 / "cation.csv", index=False)
    original_il = (stage1 / "IL.csv").read_bytes()

    augmented = augment_pretraining_entities(
        output_root,
        pretrain_cap=100,
        client=EmptyPubChemClient(),
    )

    resonance = augmented[
        augmented["rule_list"].map(
            lambda value: "resonance_equivalent" in str(value)
        )
    ]
    assert len(resonance) == 1
    assert resonance.iloc[0]["origin_list"] == "rule"
    assert resonance.iloc[0]["seed_smiles_list"] == "CCCC[n+]1ccn(C)c1"
    assert fixed_h_identity_key(resonance.iloc[0]["SMILES"]) == (
        fixed_h_identity_key("CCCC[n+]1ccn(C)c1")
    )
    assert (stage1 / "IL.csv").read_bytes() == original_il


def test_augmentation_uses_global_cap_and_records_selected_provenance(
    tmp_path: Path,
):
    output_root = tmp_path / "training"
    stage1 = output_root / "stage1"
    stage1.mkdir(parents=True)
    pd.DataFrame(columns=["cation", "anion"]).to_csv(
        stage1 / "IL.csv",
        index=False,
    )
    for role in ("anion", "cation", "solute", "solvent"):
        pd.DataFrame(columns=PRETRAIN_ENTITY_COLUMNS).to_csv(
            stage1 / f"{role}.csv",
            index=False,
        )
    pd.DataFrame(
        [
            {
                "SMILES": "CCF",
                "formal_charge": 0,
                "origin_list": "dataset",
                "seed_smiles_list": "",
                "rule_list": "",
                "pubchem_cid_list": "",
                "mol_id_list": "",
            }
        ],
        columns=PRETRAIN_ENTITY_COLUMNS,
    ).to_csv(stage1 / "molecule.csv", index=False)

    augmented = augment_pretraining_entities(
        output_root,
        pretrain_cap=4,
        client=FakePubChemClient(),
    )

    assert len(augmented) == 4
    assert set(augmented["SMILES"]) == {"CCF", "CCBr", "CCCF", "CCO"}
    molecule = pd.read_csv(stage1 / "molecule.csv", keep_default_na=False)
    assert list(molecule.columns) == PRETRAIN_ENTITY_COLUMNS
    assert molecule.set_index("SMILES").loc["CCF", "origin_list"] == "dataset"
    overlap = molecule.set_index("SMILES").loc["CCBr"]
    assert overlap["origin_list"] == "pubchem;rule"
    assert overlap["seed_smiles_list"] == "CCF"
    assert overlap["rule_list"] == "halogen_F_to_Br"
    assert str(overlap["pubchem_cid_list"]) == "2"
    local_rule = molecule.set_index("SMILES").loc["CCCF"]
    assert local_rule["origin_list"] == "rule"
    assert local_rule["pubchem_cid_list"] == ""
    pubchem_only = molecule.set_index("SMILES").loc["CCO"]
    assert pubchem_only["origin_list"] == "pubchem"
    assert str(pubchem_only["pubchem_cid_list"]) == "3"
    assert not (stage1 / "augmentation_provenance.csv").exists()


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


def test_weighted_shared_folds_improve_skewed_row_balance():
    weights = {
        "task_a": {
            f"group_{index}": weight
            for index, weight in enumerate(
                [50, 40, 30, 20, 10, 1, 1, 1, 1, 1]
            )
        },
        "task_b": {
            f"group_{index}": weight
            for index, weight in enumerate(
                [1, 1, 1, 1, 1, 10, 20, 30, 40, 50]
            )
        },
    }
    repeats = {"task_a": 1, "task_b": 1}
    baseline = plan_shared_group_folds(
        {
            task_id: set(task_weights)
            for task_id, task_weights in weights.items()
        },
        repeats,
        seed=42,
        namespace="test_weighted",
    )
    weighted = plan_shared_weighted_group_folds(
        weights,
        repeats,
        seed=42,
        namespace="test_weighted",
    )
    repeated = plan_shared_weighted_group_folds(
        weights,
        repeats,
        seed=42,
        namespace="test_weighted",
    )

    def squared_deviation(
        assignments: dict[tuple[str, int], int],
    ) -> float:
        total = 0.0
        for task_weights in weights.values():
            loads = [0, 0, 0, 0, 0]
            for group_identifier, row_count in task_weights.items():
                loads[assignments[(group_identifier, 0)]] += row_count
            assert min(loads) > 0
            mean = sum(loads) / 5.0
            total += sum((load - mean) ** 2 for load in loads)
        return total

    assert weighted == repeated
    assert squared_deviation(weighted) < squared_deviation(baseline)
    assert {
        weighted[(group_identifier, 0)]
        for group_identifier in weights["task_a"]
    } == set(range(5))


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


def _system_rows(
    rows: list[tuple[str, str, int]],
) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["_system_id", "_system_key", "row_count"],
    )


def test_joint_test_selection_avoids_smaller_tasks_and_reuses_shared_systems():
    tasks = [
        TaskSpec("large/a", 3, "a.csv", ("value",), ("SMILES",), "molecule"),
        TaskSpec("large/b", 3, "b.csv", ("value",), ("SMILES",), "molecule"),
        TaskSpec("small/c", 3, "c.csv", ("value",), ("SMILES",), "molecule"),
    ]
    profiles = {
        "large/a": TaskProfile(10, 10, 10, "large"),
        "large/b": TaskProfile(10, 10, 10, "large"),
        "small/c": TaskProfile(1, 1, 1, "small"),
    }
    systems = {
        "large/a": _system_rows(
            [
                ("shared_safe", '["shared_safe"]', 1),
                ("shared_unsafe", '["shared_unsafe"]', 1),
                ("only_a", '["only_a"]', 1),
            ]
        ),
        "large/b": _system_rows(
            [
                ("shared_safe", '["shared_safe"]', 1),
                ("shared_unsafe", '["shared_unsafe"]', 1),
                ("only_b", '["only_b"]', 1),
            ]
        ),
        "small/c": _system_rows(
            [("shared_unsafe", '["shared_unsafe"]', 1)]
        ),
    }

    selected = plan_joint_test_registry(
        tasks,
        profiles,
        systems,
        seed=42,
    )

    assert set(selected) == {("molecule", "shared_safe")}
    assert selected[("molecule", "shared_safe")]["source_tasks"] == {
        "large/a",
        "large/b",
    }
    assert not selected[("molecule", "shared_safe")]["reserved_tasks"]


def test_joint_test_selection_fallback_minimizes_reserved_systems_then_rows():
    tasks = [
        TaskSpec("large/a", 3, "a.csv", ("value",), ("SMILES",), "molecule"),
        TaskSpec("small/a", 3, "s1.csv", ("value",), ("SMILES",), "molecule"),
        TaskSpec("small/b", 3, "s2.csv", ("value",), ("SMILES",), "molecule"),
    ]
    profiles = {
        "large/a": TaskProfile(10, 10, 10, "large"),
        "small/a": TaskProfile(10, 10, 10, "small"),
        "small/b": TaskProfile(10, 10, 10, "small"),
    }
    systems = {
        "large/a": _system_rows(
            [
                ("one_task_many_rows", '["one_task_many_rows"]', 1),
                ("one_task_few_rows", '["one_task_few_rows"]', 1),
                ("two_tasks", '["two_tasks"]', 1),
            ]
        ),
        "small/a": _system_rows(
            [
                ("one_task_many_rows", '["one_task_many_rows"]', 20),
                ("one_task_few_rows", '["one_task_few_rows"]', 2),
                ("two_tasks", '["two_tasks"]', 1),
            ]
        ),
        "small/b": _system_rows(
            [("two_tasks", '["two_tasks"]', 1)]
        ),
    }

    selected = plan_joint_test_registry(
        tasks,
        profiles,
        systems,
        seed=42,
    )

    assert set(selected) == {("molecule", "one_task_few_rows")}


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

    small_indices = list(range(1, 50))
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

    density_root = output_root / "stage3" / "experiment" / "density"
    conductivity_root = (
        output_root
        / "stage3"
        / "experiment"
        / "electrical_conductivity"
    )
    density_test = pd.read_csv(density_root / "test.csv")
    conductivity_test = pd.read_csv(conductivity_root / "test.csv")
    density_test_systems = set(
        density_test[["cation", "anion"]].itertuples(index=False, name=None)
    )
    conductivity_test_systems = set(
        conductivity_test[["cation", "anion"]].itertuples(
            index=False,
            name=None,
        )
    )
    assert len(density_test_systems) == 50
    assert density_test_systems == conductivity_test_systems
    small_systems = {
        ion_pair(index)
        for index in small_indices
    }
    assert density_test_systems.isdisjoint(small_systems)

    density_folds = [
        pd.read_csv(density_root / "random" / f"fold{fold}.csv")
        for fold in range(1, 6)
    ]
    assert sum(map(len, density_folds)) + len(density_test) == 501
    density_development_systems = {
        pair
        for fold in density_folds
        for pair in fold[["cation", "anion"]].itertuples(
            index=False,
            name=None,
        )
    }
    assert density_test_systems.isdisjoint(density_development_systems)

    small_root = (
        output_root
        / "stage3"
        / "experiment"
        / "dynamic_relative_permittivity"
    )
    reserved = pd.read_csv(small_root / "reserved_summary.csv")
    assert reserved.empty
    assert len(pd.read_csv(small_root / "loo_manifest.csv")) == 49
    for repeat in range(1, 6):
        folds = [
            pd.read_csv(
                small_root
                / "random"
                / f"cv{repeat}"
                / f"fold{fold}.csv"
            )
            for fold in range(1, 6)
        ]
        assert sum(map(len, folds)) == 49

    medium_root = output_root / "stage3" / "experiment" / "solvation"
    assert (
        medium_root / "random" / "cv5" / "fold5.csv"
    ).exists()
    balance = pd.read_csv(output_root / "_audit" / "fold_balance.csv")
    assert list(balance.columns) == [
        "task_id",
        "strategy",
        "cv",
        "fold",
        "row_count",
        "group_count",
        "total_rows",
        "mean_rows",
        "largest_group_rows",
        "max_to_min",
        "theoretical_lower_bound",
        "unavoidable_group_dominance",
    ]
    assert set(balance["strategy"]) == {"cation", "anion"}
    assert balance["row_count"].gt(0).all()

    stage2_density = output_root / "stage2" / "density"
    train = pd.read_csv(stage2_density / "train.csv")
    valid = pd.read_csv(stage2_density / "valid.csv")
    assert set(train.columns) == set(valid.columns)
    assert len(train) + len(valid) == 100
    assert set(
        train[["cation", "anion"]].itertuples(index=False, name=None)
    ).isdisjoint(
        set(valid[["cation", "anion"]].itertuples(index=False, name=None))
    )

    legacy_paths = [
        output_root / "task_catalog.csv",
        output_root / "manifest.json",
        output_root / "audit",
        output_root / "stage2" / "rows.csv",
        output_root / "stage3" / "rows.csv",
        output_root / "stage3" / "test_groups.csv",
    ]
    assert not any(path.exists() for path in legacy_paths)
    for path in output_root.rglob("*.csv"):
        assert not {
            "row_id",
            "system_id",
            "entity_id",
        } & set(pd.read_csv(path, nrows=0).columns)
    assert ignored_structure.exists()

    checksummed_paths = [
        output_root / "stage1" / "IL.csv",
        output_root / "stage2" / "density" / "train.csv",
        density_root / "test.csv",
        density_root / "IL" / "fold1.csv",
        small_root / "random" / "cv3" / "fold4.csv",
    ]
    first_run = {path: path.read_bytes() for path in checksummed_paths}
    build_training_splits(final_root, output_root, seed=42)
    assert first_run == {path: path.read_bytes() for path in checksummed_paths}
