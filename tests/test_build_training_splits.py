import gzip
import json
from http.client import IncompleteRead
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
from rdkit import Chem

import scripts.build_training_splits as training_splits
from scripts.build_training_splits import (
    IncompletePubChemQuery,
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
    discover_tasks,
    extract_pretraining_entities,
    fixed_h_identity_key,
    formal_charge,
    fragment_count,
    generate_rule_candidates,
    import_zinc_diversity,
    ionize_zinc_parent,
    plan_joint_test_registry,
    prepare_task_frame,
    resonance_eligibility,
    role_fragments,
    task_group_kfold_assignments,
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


def canonical_ion_pair(index: int) -> tuple[str, str]:
    """Return the canonicalized ion pair generated for an index."""
    return tuple(canonicalize_smiles(value) for value in ion_pair(index))


def neutral_carbon(index: int) -> str:
    return "C" * index


def neutral_oxygen(index: int) -> str:
    return "O" + ("C" * index)


def canonical_transfer_system(index: int) -> tuple[str, str]:
    """Return the canonicalized transfer system generated for an index."""
    return (
        canonicalize_smiles(neutral_carbon(index)),
        canonicalize_smiles(neutral_oxygen(index)),
    )


def write_stage2_files(final_root: Path, count: int = 100) -> None:
    pairs = [ion_pair(index) for index in range(1, count + 1)]
    for filename, target in (
        ("density.csv", "density_g/cm^3"),
        ("heat_capacity.csv", "heat_capacity_J/mol/K"),
        ("thermal_expansion.csv", "thermal_expansion_K^-1"),
    ):
        rows = [
            {
                "cation": cation,
                "anion": anion,
                "temperature_K": 298.0,
                target: float(index),
                "source_list": "simulation",
            }
            for index, (cation, anion) in enumerate(pairs)
        ]
        rows.append(
            {
                "cation": pairs[0][0],
                "anion": pairs[0][1],
                "temperature_K": 320.0,
                target: -1.0,
                "source_list": "simulation",
            }
        )
        write_csv(
            final_root / "simulation" / filename,
            rows,
        )
    qm_rows = [
        {
            "SMILES": neutral_carbon(index),
            **{column: float(index) for column in QM_COLUMNS},
            "source_list": "simulation",
        }
        for index in range(1, count + 1)
    ]
    qm_rows.append(qm_rows[0].copy())
    write_csv(
        final_root / "simulation" / "simulated_qm_elec_hf.csv",
        qm_rows,
    )
    transfer_rows = [
        {
            "solute": neutral_carbon(index),
            "solvent": neutral_oxygen(index),
            "transfer_organic_kcal/mol": float(index),
            "source_list": "simulation",
        }
        for index in range(1, count + 1)
    ]
    transfer_rows.append(transfer_rows[0].copy())
    write_csv(
        final_root / "simulation" / "transfer_organic.csv",
        transfer_rows,
    )
    write_csv(
        final_root / "simulation" / "heat_of_vaporization.csv",
        [
            {
                "cation": cation,
                "anion": anion,
                "temperature_K": 298.0,
                "heat_of_vaporization_kJ/mol": float(index),
                "source_list": "simulation",
            }
            for index, (cation, anion) in enumerate(pairs)
        ]
        + [
            {
                "cation": pairs[0][0],
                "anion": pairs[0][1],
                "temperature_K": 320.0,
                "heat_of_vaporization_kJ/mol": -1.0,
                "source_list": "simulation",
            }
        ],
    )
    write_csv(
        final_root / "simulation" / "pbe_tzvp_cation_orbitals.csv",
        [
            {
                "cation": cation,
                "HOMO_eV": -float(index + 2),
                "LUMO_eV": -float(index),
                "source_list": "simulation",
            }
            for index, (cation, _anion) in enumerate(pairs)
        ],
    )
    write_csv(
        final_root / "simulation" / "pbe_tzvp_anion_orbitals.csv",
        [
            {
                "anion": anion,
                "HOMO_eV": -float(index + 2),
                "LUMO_eV": -float(index),
                "source_list": "simulation",
            }
            for index, (_cation, anion) in enumerate(pairs)
        ],
    )
    charge_rows = [
        {
            "mol_id": f"mol_{index:07d}",
            "SMILES": neutral_carbon(index),
            "charge": 0,
            "source_list": "simulation",
        }
        for index in range(1, count + 1)
    ]
    charge_rows.append(
        {
            "mol_id": "mol_duplicate",
            "SMILES": neutral_carbon(1),
            "charge": 0,
            "source_list": "simulation",
        }
    )
    write_csv(final_root / "simulation" / "charge.csv", charge_rows)
    write_csv(
        final_root
        / "simulation"
        / "charge_20260514"
        / "structure_manifest.csv",
        [
            {
                "mol_id": row["mol_id"],
                "relative_path": f"{row['mol_id']}.mol2",
                "format": "mol2",
                "size_bytes": 1,
                "sha256": "0" * 64,
                "referenced_by_charge": True,
            }
            for row in charge_rows
        ],
    )


def test_extract_pretraining_entities_splits_and_normalizes_roles_by_charge(
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
                "mol_id": "solute_id",
                "temperature_K": 298.15,
                "solvation_kcal/mol": -2.0,
                "source_list": "test",
            },
            {
                "cation": "[Na+].[K+]",
                "anion": "[Cl-].[Br-]",
                "solute": "CC(=O)[O-]",
                "temperature_K": 298.15,
                "solvation_kcal/mol": -3.0,
                "source_list": "test",
            },
        ],
    )
    write_csv(
        final_root / "experiment" / "density.csv",
        [
            {
                "cation": "C[NH3+]",
                "anion": "CC(=O)[O-]",
                "density_g/cm^3": 1.0,
                "source_list": "experiment",
            }
        ],
    )
    write_csv(
        final_root / "simulation" / "molecules.csv",
        [
            {
                "SMILES": "OCC",
                "mol_id": "simulation_id",
                "value": 1.0,
                "source_list": "test",
            },
            {
                "SMILES": "CC(=O)[O-]",
                "mol_id": "negative_molecule",
                "value": 2.0,
                "source_list": "test",
            },
            {
                "SMILES": "C[NH3+]",
                "mol_id": "positive_molecule",
                "value": 3.0,
                "source_list": "test",
            },
        ],
    )
    write_csv(
        final_root / "simulation" / "density.csv",
        [
            {
                "cation": "[Na+].[K+]",
                "anion": "[Cl-].[Br-]",
                "density_g/cm^3": 1.0,
                "source_list": "simulation",
            },
            {
                "cation": "[Na+]",
                "anion": "[Cl-]",
                "density_g/cm^3": 1.1,
                "source_list": "simulation",
            },
        ],
    )
    structure_path = final_root / "simulation" / "charge_20260514" / "bad.mol"
    structure_path.parent.mkdir(parents=True)
    structure_path.write_text("not a molecule", encoding="utf-8")

    output_root = tmp_path / "training"
    stale_augmentation = output_root / "stage1" / "augmentation"
    stale_augmentation.mkdir(parents=True)
    (stale_augmentation / "stale.txt").write_text(
        "stale\n",
        encoding="utf-8",
    )
    entities = extract_pretraining_entities(final_root, output_root)

    assert set(entities["role"]) == {"cation", "anion", "molecule"}
    assert set(entities.loc[entities["role"].eq("cation"), "SMILES"]) == {
        "C[NH3+]",
        "[K+]",
        "[Na+]",
    }
    assert set(entities.loc[entities["role"].eq("anion"), "SMILES"]) == {
        "CC(=O)[O-]",
        "[Br-]",
        "[Cl-]",
    }
    ethanol = entities[entities["SMILES"].eq("CCO")]
    assert ethanol["role"].tolist() == ["molecule"]
    assert ethanol.iloc[0]["mol_id_list"] == "simulation_id;solute_id"
    acetate = entities[entities["SMILES"].eq("CC(=O)[O-]")]
    assert set(acetate["role"]) == {"anion"}
    assert acetate.iloc[0]["mol_id_list"] == "negative_molecule"
    methylammonium = entities[entities["SMILES"].eq("C[NH3+]")]
    assert set(methylammonium["role"]) == {"cation"}
    assert methylammonium.iloc[0]["mol_id_list"] == "positive_molecule"
    assert "il" not in set(entities["role"])

    assert {
        path.name
        for path in (output_root / "stage1").glob("*.csv")
    } == {
        "IL.csv",
        "experiment_IL.csv",
        "anion.csv",
        "cation.csv",
        "solute.csv",
        "solvent.csv",
        "molecule.csv",
        "simulation_mol.csv",
    }
    il = pd.read_csv(output_root / "stage1" / "IL.csv")
    assert list(il.columns) == ["cation", "anion"]
    assert il.to_dict(orient="records") == [
        {"cation": cation, "anion": anion}
        for cation, anion in sorted(
            {
                (
                    canonicalize_smiles("[Na+].[K+]"),
                    canonicalize_smiles("[Cl-].[Br-]"),
                ),
                (
                    canonicalize_smiles("C[NH3+]"),
                    canonicalize_smiles("CC(=O)[O-]"),
                ),
                (
                    canonicalize_smiles("[Na+]"),
                    canonicalize_smiles("[Cl-]"),
                ),
            }
        )
    ]
    experiment_il = pd.read_csv(
        output_root / "stage1" / "experiment_IL.csv"
    )
    assert list(experiment_il.columns) == ["cation", "anion"]
    assert experiment_il.to_dict(orient="records") == [
        {"cation": cation, "anion": anion}
        for cation, anion in sorted(
            {
                (
                    canonicalize_smiles("[Na+].[K+]"),
                    canonicalize_smiles("[Cl-].[Br-]"),
                ),
                (
                    canonicalize_smiles("C[NH3+]"),
                    canonicalize_smiles("CC(=O)[O-]"),
                ),
            }
        )
    ]
    molecule = pd.read_csv(output_root / "stage1" / "molecule.csv")
    assert list(molecule.columns) == PRETRAIN_ENTITY_COLUMNS
    assert molecule["SMILES"].tolist() == ["CCO"]
    assert molecule.loc[0, "mol_id_list"] == "simulation_id;solute_id"
    assert molecule.loc[0, "origin_list"] == "dataset"
    simulation_mol = pd.read_csv(
        output_root / "stage1" / "simulation_mol.csv"
    )
    assert simulation_mol["SMILES"].tolist() == ["CCO"]
    assert simulation_mol.loc[0, "mol_id_list"] == "simulation_id"
    solute = pd.read_csv(output_root / "stage1" / "solute.csv")
    assert solute["SMILES"].tolist() == ["CCO"]
    assert solute.loc[0, "mol_id_list"] == "solute_id"
    for filename in (
        "anion.csv",
        "cation.csv",
        "molecule.csv",
        "simulation_mol.csv",
        "solute.csv",
        "solvent.csv",
    ):
        frame = pd.read_csv(
            output_root / "stage1" / filename,
            keep_default_na=False,
        )
        assert set(frame["origin_list"]) <= {"dataset"}
        assert not frame[
            ["seed_smiles_list", "rule_list", "pubchem_cid_list"]
        ].astype(bool).any(axis=None)
    assert not (output_root / "stage1" / "entities.csv").exists()
    assert not (output_root / "stage1" / "entity_sources.csv").exists()
    assert not (output_root / "stage1" / "augmentation").exists()
    assert structure_path.exists()


def test_extract_pretraining_entities_writes_empty_experiment_il(
    tmp_path: Path,
):
    final_root = tmp_path / "final"
    write_csv(
        final_root / "experiment" / "molecules.csv",
        [{"SMILES": "CCO", "source_list": "experiment"}],
    )
    write_csv(
        final_root / "simulation" / "density.csv",
        [
            {
                "cation": "C[NH3+]",
                "anion": "CC(=O)[O-]",
                "density_g/cm^3": 1.1,
                "source_list": "simulation",
            }
        ],
    )

    output_root = tmp_path / "training"
    extract_pretraining_entities(final_root, output_root)

    experiment_il = pd.read_csv(
        output_root / "stage1" / "experiment_IL.csv"
    )
    assert list(experiment_il.columns) == ["cation", "anion"]
    assert experiment_il.empty
    il = pd.read_csv(output_root / "stage1" / "IL.csv")
    assert len(il) == 1


def test_rule_candidates_apply_shared_rules_to_neutral_entities():
    alkyl = generate_rule_candidates("CCCC", "molecule")
    assert any(row["rule"] == "terminal_alkyl_extend_1" for row in alkyl)
    assert any(
        row["rule"] == "terminal_alkyl_linear_to_branch"
        for row in alkyl
    )

    halogen = generate_rule_candidates("CCF", "molecule")
    assert any(row["rule"] == "halogen_F_to_Cl" for row in halogen)

    neutral_resonance = generate_rule_candidates(
        "N#[N+][O-]",
        "molecule",
    )
    assert {
        (row["rule"], row["SMILES"])
        for row in neutral_resonance
    } == {
        ("resonance_equivalent", "[N-]=[N+]=O"),
    }

    charged = generate_rule_candidates("C[NH3+]", "molecule")
    assert charged
    assert all(candidate_allowed("C[NH3+]", row["SMILES"], "cation") for row in charged)
    assert any(
        row["rule"] == "terminal_alkyl_extend_4"
        for row in charged
    )
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

    neutral = {
        row["rule"]
        for row in generate_rule_candidates("CCCC", "molecule")
    }
    assert "terminal_alkyl_extend_4" in neutral
    assert "terminal_alkyl_linear_to_branch" in neutral


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
    assert any(
        row["rule"] == "cation_headgroup_N_to_P"
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
    assert any(
        row["rule"] == "halogen_Cl_to_I"
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


class AllCandidatePubChemClient:
    def similarity(
        self,
        smiles: str,
        *,
        threshold: int,
        max_records: int,
    ) -> list[dict[str, object]]:
        assert threshold == 90
        assert max_records == 100
        canonical = canonicalize_smiles(smiles)
        if canonical == "CCF":
            return [
                {"CID": 1, "SMILES": "CCF", "Charge": 0},
                {"CID": 2, "SMILES": "CCO", "Charge": 0},
                {"CID": 3, "SMILES": "CCBr", "Charge": 0},
                {"CID": 4, "SMILES": "CCC", "Charge": 0},
            ]
        assert canonical == "CCO"
        return [
            {"CID": 5, "SMILES": "CCO", "Charge": 0},
            {"CID": 6, "SMILES": "CCF", "Charge": 0},
            {"CID": 7, "SMILES": "CO", "Charge": 0},
            {"CID": 8, "SMILES": "CN", "Charge": 0},
        ]


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


class NeutralResonancePubChemClient:
    def similarity(
        self,
        smiles: str,
        *,
        threshold: int,
        max_records: int,
    ) -> list[dict[str, object]]:
        assert threshold == 90
        assert max_records == 100
        assert canonicalize_smiles(smiles) == "N#[N+][O-]"
        return [
            {"CID": 1, "SMILES": "N#[N+][O-]", "Charge": 0},
            {"CID": 2, "SMILES": "[N-]=[N+]=O", "Charge": 0},
        ]

    def identity(self, _smiles: str) -> list[dict[str, object]]:
        raise AssertionError("resonance rules must not call PubChem identity")


def write_stage1_molecules(
    output_root: Path,
    smiles_values: list[str],
) -> bytes:
    stage1 = output_root / "stage1"
    stage1.mkdir(parents=True)
    pd.DataFrame(columns=["cation", "anion"]).to_csv(
        stage1 / "IL.csv",
        index=False,
    )
    for role in (
        "anion",
        "cation",
        "simulation_mol",
        "solute",
        "solvent",
    ):
        pd.DataFrame(columns=PRETRAIN_ENTITY_COLUMNS).to_csv(
            stage1 / f"{role}.csv",
            index=False,
        )
    pd.DataFrame(
        [
            {
                "SMILES": smiles,
                "formal_charge": formal_charge(smiles),
                "origin_list": "dataset",
                "seed_smiles_list": "",
                "rule_list": "",
                "pubchem_cid_list": "",
                "mol_id_list": "",
            }
            for smiles in smiles_values
        ],
        columns=PRETRAIN_ENTITY_COLUMNS,
    ).to_csv(stage1 / "molecule.csv", index=False)
    return (stage1 / "molecule.csv").read_bytes()


def write_existing_augmentation(output_root: Path) -> dict[Path, bytes]:
    augmentation = output_root / "stage1" / "augmentation"
    augmentation.mkdir()
    for role in ("anion", "cation", "molecule"):
        pd.DataFrame(columns=PRETRAIN_ENTITY_COLUMNS).to_csv(
            augmentation / f"{role}.csv",
            index=False,
        )
    marker = augmentation / "marker.txt"
    marker.write_text("previous augmentation\n", encoding="utf-8")
    return {
        path: path.read_bytes()
        for path in augmentation.iterdir()
        if path.is_file()
    }


def write_zinc_shard(
    path: Path,
    rows: list[tuple[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        for smiles, zinc_id in rows:
            handle.write(f"{smiles}\t{zinc_id}\n")


def write_zinc_manifest(root: Path, relative_paths: list[str]) -> Path:
    manifest = root.parent / "zinc22_smi_urls.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(
            f"https://cache.docking.org/zinc22/{relative_path}\n"
            for relative_path in relative_paths
        ),
        encoding="utf-8",
    )
    return manifest


def write_zinc_failed_log(
    root: Path,
    relative_paths: list[str],
    *,
    extra_urls: list[str] | None = None,
) -> Path:
    failed_log = root.parent / "zinc22_smi_failed.log"
    failed_log.write_text(
        "".join(
            [
                *(
                    f"https://cache.docking.org/zinc22/{relative_path}\n"
                    for relative_path in relative_paths
                ),
                *(f"{url}\n" for url in (extra_urls or [])),
            ]
        ),
        encoding="utf-8",
    )
    return failed_log


def write_empty_stage1(output_root: Path) -> None:
    write_stage1_molecules(output_root, [])
    write_existing_augmentation(output_root)


def test_zinc_ionization_rules_are_conservative_and_deterministic():
    carboxylate = ionize_zinc_parent("CC(=O)O", "anion")
    ammonium = ionize_zinc_parent("CCCCN", "cation")
    pyridinium = ionize_zinc_parent("c1ccncc1", "cation")

    assert carboxylate is not None
    assert carboxylate["rule"] == "zinc_deprotonate_carboxylic_acid"
    assert formal_charge(str(carboxylate["SMILES"])) == -1
    assert ammonium is not None
    assert ammonium["rule"] == "zinc_protonate_amine"
    assert formal_charge(str(ammonium["SMILES"])) == 1
    assert pyridinium is not None
    assert "[nH+]" in str(pyridinium["SMILES"])

    assert ionize_zinc_parent("CCCC(=O)N", "cation") is None
    first = ionize_zinc_parent("OC(=O)CCC(=O)O", "anion")
    second = ionize_zinc_parent("O=C(O)CCC(O)=O", "anion")
    assert first is not None and second is not None
    assert first["SMILES"] == second["SMILES"]
    for result in (carboxylate, ammonium, pyridinium, first):
        smiles = str(result["SMILES"])
        assert fragment_count(smiles) == 1
        assert formal_charge(smiles) in {-1, 1}


def test_zinc_scan_only_freezes_startup_snapshot(
    tmp_path: Path,
    monkeypatch,
):
    output_root = tmp_path / "training"
    zinc_root = tmp_path / "ZINC" / "zinc22_smi_data"
    first_path = zinc_root / "zinc-22a/H06/H06M000/H06M000-M-a.smi.gz"
    second_path = zinc_root / "zinc-22a/H07/H07O000/H07O000-O-a.smi.gz"
    write_zinc_shard(first_path, [("CC(=O)O", "ZINC0001")])
    manifest = write_zinc_manifest(
        zinc_root,
        [
            first_path.relative_to(zinc_root).as_posix(),
            second_path.relative_to(zinc_root).as_posix(),
        ],
    )
    write_empty_stage1(output_root)

    original_hash = training_splits.file_sha256

    def add_file_after_snapshot(path: Path) -> str:
        digest = original_hash(path)
        if path == first_path and not second_path.exists():
            write_zinc_shard(second_path, [("CCCCN", "ZINC0002")])
        return digest

    monkeypatch.setattr(training_splits, "file_sha256", add_file_after_snapshot)
    summary = import_zinc_diversity(
        output_root,
        zinc_root=zinc_root,
        zinc_manifest=manifest,
        target_per_ion_role=1,
        scan_only=True,
    )

    assert summary["snapshot_files"] == 1
    assert summary["manifest_files"] == 2
    assert summary["processed_files"] == 1
    assert (output_root / "stage1" / "augmentation" / "marker.txt").exists()

    resumed = import_zinc_diversity(
        output_root,
        zinc_root=zinc_root,
        zinc_manifest=manifest,
        target_per_ion_role=1,
        scan_only=True,
    )
    assert resumed["snapshot_files"] == 2
    assert resumed["processed_files"] == 1


def test_zinc_formal_incomplete_input_preserves_augmentation(tmp_path: Path):
    output_root = tmp_path / "training"
    zinc_root = tmp_path / "ZINC" / "zinc22_smi_data"
    path = zinc_root / "zinc-22a/H06/H06M000/H06M000-M-a.smi.gz"
    manifest = write_zinc_manifest(
        zinc_root,
        [path.relative_to(zinc_root).as_posix()],
    )
    write_empty_stage1(output_root)
    before = {
        item.relative_to(output_root): item.read_bytes()
        for item in (output_root / "stage1" / "augmentation").rglob("*")
        if item.is_file()
    }

    with pytest.raises(TrainingSplitError, match="incomplete"):
        import_zinc_diversity(
            output_root,
            zinc_root=zinc_root,
            zinc_manifest=manifest,
            target_per_ion_role=1,
        )

    assert before == {
        item.relative_to(output_root): item.read_bytes()
        for item in (output_root / "stage1" / "augmentation").rglob("*")
        if item.is_file()
    }


def test_zinc_formal_skips_logged_missing_and_empty_sources(tmp_path: Path):
    output_root = tmp_path / "training"
    zinc_root = tmp_path / "ZINC" / "zinc22_smi_data"
    available_paths = [
        zinc_root / "zinc-22a/H06/H06M000/H06M000-M.a.smi.gz",
        zinc_root / "zinc-22a/H06/H06O000/H06O000-O.a.smi.gz",
    ]
    missing_path = zinc_root / "zinc-22a/H07/H07M000/H07M000-M.a.smi.gz"
    empty_path = zinc_root / "zinc-22a/H07/H07O000/H07O000-O.a.smi.gz"
    write_zinc_shard(available_paths[0], [("CC(=O)O", "ZINC0001")])
    write_zinc_shard(available_paths[1], [("CCCCN", "ZINC0002")])
    empty_path.parent.mkdir(parents=True)
    empty_path.touch()
    relative_paths = [
        path.relative_to(zinc_root).as_posix()
        for path in [*available_paths, missing_path, empty_path]
    ]
    manifest = write_zinc_manifest(zinc_root, relative_paths)
    failed_log = write_zinc_failed_log(
        zinc_root,
        [relative_paths[2], relative_paths[3], relative_paths[2]],
        extra_urls=["https://cache.docking.org/zinc22/not-in-manifest.smi.gz"],
    )
    # A historical failure must not override a subsequently valid local file.
    with failed_log.open("a", encoding="utf-8") as handle:
        handle.write(
            "https://cache.docking.org/zinc22/"
            f"{relative_paths[0]}\n"
        )
    write_empty_stage1(output_root)

    summary = import_zinc_diversity(
        output_root,
        zinc_root=zinc_root,
        zinc_manifest=manifest,
        zinc_failed_log=failed_log,
        minimum_per_ion_role=1,
    )

    assert summary["manifest_files"] == 4
    assert summary["available_source_files"] == 2
    assert summary["logged_unavailable_source_files"] == 2
    assert summary["failed_log_nonmanifest_urls"] == 1
    unavailable = pd.read_csv(
        output_root
        / "stage1/augmentation/_audit/zinc_unavailable_sources.csv"
    )
    assert dict(zip(unavailable["relative_path"], unavailable["reason"])) == {
        relative_paths[2]: "missing_logged_failure",
        relative_paths[3]: "empty_logged_failure",
    }


def test_zinc_logged_part_file_still_blocks_formal_import(tmp_path: Path):
    output_root = tmp_path / "training"
    zinc_root = tmp_path / "ZINC" / "zinc22_smi_data"
    path = zinc_root / "zinc-22a/H06/H06M000/H06M000-M.a.smi.gz"
    part = path.with_name(f"{path.name}.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"partial")
    relative_path = path.relative_to(zinc_root).as_posix()
    manifest = write_zinc_manifest(zinc_root, [relative_path])
    failed_log = write_zinc_failed_log(zinc_root, [relative_path])
    write_empty_stage1(output_root)

    with pytest.raises(TrainingSplitError, match=r"\.part"):
        import_zinc_diversity(
            output_root,
            zinc_root=zinc_root,
            zinc_manifest=manifest,
            zinc_failed_log=failed_log,
            minimum_per_ion_role=1,
        )


def test_zinc_minimum_allows_existing_excess_without_truncation(
    tmp_path: Path,
):
    output_root = tmp_path / "training"
    zinc_root = tmp_path / "ZINC" / "zinc22_smi_data"
    missing_paths = [
        zinc_root / "zinc-22a/H06/H06M000/H06M000-M.a.smi.gz",
        zinc_root / "zinc-22a/H06/H06O000/H06O000-O.a.smi.gz",
    ]
    relative_paths = [
        path.relative_to(zinc_root).as_posix() for path in missing_paths
    ]
    manifest = write_zinc_manifest(zinc_root, relative_paths)
    failed_log = write_zinc_failed_log(zinc_root, relative_paths)
    write_empty_stage1(output_root)
    augmentation = output_root / "stage1" / "augmentation"
    ions = {
        "anion": ["CC(=O)[O-]", "CCC(=O)[O-]"],
        "cation": ["CCC[NH3+]", "CCCC[NH3+]"],
    }
    for role, smiles_values in ions.items():
        pd.DataFrame(
            [
                {
                    "SMILES": canonicalize_smiles(smiles),
                    "formal_charge": formal_charge(smiles),
                    "origin_list": "rule",
                    "seed_smiles_list": "seed",
                    "rule_list": "existing_rule",
                    "pubchem_cid_list": "",
                    "mol_id_list": "",
                }
                for smiles in smiles_values
            ],
            columns=PRETRAIN_ENTITY_COLUMNS,
        ).to_csv(augmentation / f"{role}.csv", index=False)

    summary = import_zinc_diversity(
        output_root,
        zinc_root=zinc_root,
        zinc_manifest=manifest,
        zinc_failed_log=failed_log,
        minimum_per_ion_role=1,
    )

    assert summary["final_unique_by_role"] == {"anion": 2, "cation": 2}
    assert summary["minimum_met_by_role"] == {"anion": True, "cation": True}
    assert summary["excess_by_role"] == {"anion": 1, "cation": 1}
    for role in ions:
        assert len(pd.read_csv(augmentation / f"{role}.csv")) == 2


def test_zinc_minimum_publishes_all_eligible_candidates(
    tmp_path: Path,
):
    output_root = tmp_path / "training"
    zinc_root = tmp_path / "ZINC" / "zinc22_smi_data"
    paths = [
        zinc_root / "zinc-22a/H06/H06M000/H06M000-M.a.smi.gz",
        zinc_root / "zinc-22a/H06/H06O000/H06O000-O.a.smi.gz",
    ]
    write_zinc_shard(
        paths[0],
        [
            ("CC(=O)O", "ZINC0001"),
            ("CCC(=O)O", "ZINC0002"),
        ],
    )
    write_zinc_shard(
        paths[1],
        [
            ("CCCN", "ZINC0003"),
            ("CCCCN", "ZINC0004"),
        ],
    )
    manifest = write_zinc_manifest(
        zinc_root,
        [path.relative_to(zinc_root).as_posix() for path in paths],
    )
    write_empty_stage1(output_root)

    summary = import_zinc_diversity(
        output_root,
        zinc_root=zinc_root,
        zinc_manifest=manifest,
        minimum_per_ion_role=1,
    )

    assert summary["selection_mode"] == "all_eligible"
    assert summary["eligible_unique_by_role"] == {
        "anion": 2,
        "cation": 2,
    }
    assert summary["selected_new_by_role"] == {
        "anion": 2,
        "cation": 2,
    }
    assert summary["eligible_not_selected"] == 0
    assert summary["final_unique_by_role"] == {
        "anion": 2,
        "cation": 2,
    }
    assert summary["excess_by_role"] == {"anion": 1, "cation": 1}
    balance = pd.read_csv(
        output_root
        / "stage1/augmentation/_audit/zinc_selection_balance.csv"
    )
    assert balance["available"].equals(balance["quota"])
    assert balance["available"].equals(balance["selected"])


def test_zinc_import_is_order_independent_and_merges_overlap_provenance(
    tmp_path: Path,
):
    outputs: list[dict[str, list[str]]] = []
    rows_by_charge = {
        "M": [
            ("CC(=O)O", "ZINC0001"),
            ("CCC(=O)O", "ZINC0002"),
        ],
        "O": [
            ("CCCCN", "ZINC0003"),
            ("CCCN", "ZINC0004"),
        ],
    }
    for run, reverse in enumerate((False, True)):
        output_root = tmp_path / f"training-{run}"
        zinc_root = tmp_path / f"ZINC-{run}" / "zinc22_smi_data"
        relative_paths: list[str] = []
        for charge in ("M", "O"):
            path = (
                zinc_root
                / "zinc-22a/H06"
                / f"H06{charge}000"
                / f"H06{charge}000-{charge}-a.smi.gz"
            )
            rows = (
                list(reversed(rows_by_charge[charge]))
                if reverse
                else rows_by_charge[charge]
            )
            write_zinc_shard(path, rows)
            relative_paths.append(path.relative_to(zinc_root).as_posix())
        manifest = write_zinc_manifest(
            zinc_root,
            list(reversed(relative_paths)) if reverse else relative_paths,
        )
        write_empty_stage1(output_root)

        overlapping = ionize_zinc_parent("CC(=O)O", "anion")
        assert overlapping is not None
        augmentation = output_root / "stage1" / "augmentation"
        pd.DataFrame(
            [
                {
                    "SMILES": overlapping["SMILES"],
                    "formal_charge": -1,
                    "origin_list": "rule",
                    "seed_smiles_list": "seed",
                    "rule_list": "existing_rule",
                    "pubchem_cid_list": "",
                    "mol_id_list": "",
                }
            ],
            columns=PRETRAIN_ENTITY_COLUMNS,
        ).to_csv(augmentation / "anion.csv", index=False)

        summary = import_zinc_diversity(
            output_root,
            zinc_root=zinc_root,
            zinc_manifest=manifest,
            target_per_ion_role=2,
            diversity_seed=42,
        )
        assert summary["completion_status"] == "complete"
        anion = pd.read_csv(augmentation / "anion.csv", keep_default_na=False)
        overlap_row = anion.loc[anion["SMILES"].eq(overlapping["SMILES"])].iloc[0]
        assert overlap_row["origin_list"] == "rule;zinc"
        outputs.append(
            {
                role: pd.read_csv(
                    augmentation / f"{role}.csv",
                    keep_default_na=False,
                )["SMILES"].tolist()
                for role in ("anion", "cation")
            }
        )

    assert outputs[0] == outputs[1]


def test_zinc_bad_gzip_and_unknown_charge_code_are_rejected(tmp_path: Path):
    output_root = tmp_path / "training"
    write_empty_stage1(output_root)
    zinc_root = tmp_path / "ZINC" / "zinc22_smi_data"
    bad = zinc_root / "zinc-22a/H06/H06M000/H06M000-M.a.smi.gz"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a gzip stream")
    manifest = write_zinc_manifest(
        zinc_root,
        [bad.relative_to(zinc_root).as_posix()],
    )
    before = (output_root / "stage1" / "augmentation" / "marker.txt").read_bytes()

    with pytest.raises(TrainingSplitError, match="gzip"):
        import_zinc_diversity(
            output_root,
            zinc_root=zinc_root,
            zinc_manifest=manifest,
            target_per_ion_role=1,
        )
    assert (
        output_root / "stage1" / "augmentation" / "marker.txt"
    ).read_bytes() == before

    unknown = zinc_root / "zinc-22a/H06/H06M000/H06M000-X.a.smi.gz"
    write_zinc_shard(unknown, [("CC(=O)O", "ZINC0001")])
    unknown_manifest = write_zinc_manifest(
        zinc_root,
        [unknown.relative_to(zinc_root).as_posix()],
    )
    with pytest.raises(TrainingSplitError, match="charge code"):
        import_zinc_diversity(
            output_root,
            zinc_root=zinc_root,
            zinc_manifest=unknown_manifest,
            target_per_ion_role=1,
            scan_only=True,
        )


def test_zinc_bad_rows_and_duplicate_ids_have_bounded_audit(tmp_path: Path):
    output_root = tmp_path / "training"
    zinc_root = tmp_path / "ZINC" / "zinc22_smi_data"
    anion_path = zinc_root / "zinc-22a/H06/H06M000/H06M000-M.a.smi.gz"
    cation_path = zinc_root / "zinc-22a/H06/H06O000/H06O000-O.a.smi.gz"
    write_zinc_shard(
        anion_path,
        [
            ("CC(=O)O", "ZINC0001"),
            ("CCC(=O)O", "ZINC0001"),
            ("bad", "row\textra"),
        ],
    )
    write_zinc_shard(cation_path, [("CCCCN", "ZINC0002")])
    manifest = write_zinc_manifest(
        zinc_root,
        [
            anion_path.relative_to(zinc_root).as_posix(),
            cation_path.relative_to(zinc_root).as_posix(),
        ],
    )
    write_empty_stage1(output_root)

    import_zinc_diversity(
        output_root,
        zinc_root=zinc_root,
        zinc_manifest=manifest,
        target_per_ion_role=1,
    )

    audit = pd.read_csv(
        output_root
        / "stage1/augmentation/_audit/zinc_rejection_counts.csv"
    )
    counts = dict(zip(audit["reason"], audit["count"], strict=True))
    assert counts["duplicate_zinc_row"] == 1
    assert counts["malformed_row"] == 1
    samples = pd.read_csv(
        output_root
        / "stage1/augmentation/_audit/zinc_rejection_samples.csv"
    )
    assert len(samples) <= 2 * training_splits.ZINC_REJECTION_SAMPLE_LIMIT


def test_zinc_rerun_invalidates_only_changed_shard_and_is_idempotent(
    tmp_path: Path,
):
    output_root = tmp_path / "training"
    zinc_root = tmp_path / "ZINC" / "zinc22_smi_data"
    anion_path = zinc_root / "zinc-22a/H06/H06M000/H06M000-M.a.smi.gz"
    cation_path = zinc_root / "zinc-22a/H06/H06O000/H06O000-O.a.smi.gz"
    write_zinc_shard(anion_path, [("CC(=O)O", "ZINC0001")])
    write_zinc_shard(cation_path, [("CCCCN", "ZINC0002")])
    manifest = write_zinc_manifest(
        zinc_root,
        [
            anion_path.relative_to(zinc_root).as_posix(),
            cation_path.relative_to(zinc_root).as_posix(),
        ],
    )
    write_empty_stage1(output_root)
    first = import_zinc_diversity(
        output_root,
        zinc_root=zinc_root,
        zinc_manifest=manifest,
        target_per_ion_role=1,
    )
    first_anion = pd.read_csv(
        output_root / "stage1/augmentation/anion.csv"
    )["SMILES"].tolist()
    assert first["processed_files"] == 2

    write_zinc_shard(anion_path, [("CCC(=O)O", "ZINC0003")])
    second = import_zinc_diversity(
        output_root,
        zinc_root=zinc_root,
        zinc_manifest=manifest,
        target_per_ion_role=1,
    )
    second_anion = pd.read_csv(
        output_root / "stage1/augmentation/anion.csv"
    )["SMILES"].tolist()

    assert second["processed_files"] == 1
    assert first_anion != second_anion
    third = import_zinc_diversity(
        output_root,
        zinc_root=zinc_root,
        zinc_manifest=manifest,
        target_per_ion_role=1,
    )
    third_anion = pd.read_csv(
        output_root / "stage1/augmentation/anion.csv"
    )["SMILES"].tolist()
    assert third["processed_files"] == 0
    assert third_anion == second_anion
    with pytest.raises(TrainingSplitError, match="already published"):
        augment_pretraining_entities(
            output_root,
            client=EmptyPubChemClient(),
        )


@pytest.mark.parametrize(
    "minimum_option",
    ["--minimum-per-ion-role", "--target-per-ion-role"],
)
def test_zinc_cli_smoke_uses_synthetic_gzip_and_temporary_output(
    tmp_path: Path,
    minimum_option: str,
):
    output_root = tmp_path / "training"
    zinc_root = tmp_path / "ZINC" / "zinc22_smi_data"
    paths = [
        zinc_root / "zinc-22a/H06/H06M000/H06M000-M.a.smi.gz",
        zinc_root / "zinc-22a/H06/H06O000/H06O000-O.a.smi.gz",
    ]
    write_zinc_shard(paths[0], [("CC(=O)O", "ZINC0001")])
    write_zinc_shard(paths[1], [("CCCCN", "ZINC0002")])
    manifest = write_zinc_manifest(
        zinc_root,
        [path.relative_to(zinc_root).as_posix() for path in paths],
    )
    write_empty_stage1(output_root)

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(training_splits.__file__).resolve()),
            "import-zinc-diversity",
            "--output-root",
            str(output_root),
            "--zinc-root",
            str(zinc_root),
            "--zinc-manifest",
            str(manifest),
            minimum_option,
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "anion=1, cation=1" in completed.stdout
    for role, expected_charge in (("anion", -1), ("cation", 1)):
        frame = pd.read_csv(
            output_root / "stage1" / "augmentation" / f"{role}.csv"
        )
        assert len(frame) == 1
        assert formal_charge(frame.loc[0, "SMILES"]) == expected_charge


def test_zinc_scan_resumes_after_file_boundary_interruption(
    tmp_path: Path,
    monkeypatch,
):
    output_root = tmp_path / "training"
    zinc_root = tmp_path / "ZINC" / "zinc22_smi_data"
    paths = [
        zinc_root / "zinc-22a/H06/H06M000/H06M000-M.a.smi.gz",
        zinc_root / "zinc-22a/H06/H06O000/H06O000-O.a.smi.gz",
    ]
    write_zinc_shard(paths[0], [("CC(=O)O", "ZINC0001")])
    write_zinc_shard(paths[1], [("CCCCN", "ZINC0002")])
    manifest = write_zinc_manifest(
        zinc_root,
        [path.relative_to(zinc_root).as_posix() for path in paths],
    )
    write_empty_stage1(output_root)
    original_process = training_splits._process_zinc_source
    calls = 0

    def interrupt_second_file(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original_process(*args, **kwargs)

    monkeypatch.setattr(
        training_splits,
        "_process_zinc_source",
        interrupt_second_file,
    )
    with pytest.raises(KeyboardInterrupt):
        import_zinc_diversity(
            output_root,
            zinc_root=zinc_root,
            zinc_manifest=manifest,
            target_per_ion_role=1,
            scan_only=True,
        )
    monkeypatch.setattr(
        training_splits,
        "_process_zinc_source",
        original_process,
    )

    resumed = import_zinc_diversity(
        output_root,
        zinc_root=zinc_root,
        zinc_manifest=manifest,
        target_per_ion_role=1,
        scan_only=True,
    )

    assert resumed["processed_files"] == 1
    assert (output_root / "stage1/augmentation/marker.txt").exists()


def test_zinc_chemical_shortage_preserves_previous_augmentation(
    tmp_path: Path,
):
    output_root = tmp_path / "training"
    zinc_root = tmp_path / "ZINC" / "zinc22_smi_data"
    paths = [
        zinc_root / "zinc-22a/H06/H06M000/H06M000-M.a.smi.gz",
        zinc_root / "zinc-22a/H06/H06O000/H06O000-O.a.smi.gz",
    ]
    write_zinc_shard(paths[0], [("CCCC", "ZINC0001")])
    write_zinc_shard(paths[1], [("CCCCN", "ZINC0002")])
    manifest = write_zinc_manifest(
        zinc_root,
        [path.relative_to(zinc_root).as_posix() for path in paths],
    )
    write_empty_stage1(output_root)
    before = {
        path.relative_to(output_root): path.read_bytes()
        for path in (output_root / "stage1/augmentation").rglob("*")
        if path.is_file()
    }

    with pytest.raises(TrainingSplitError, match="insufficient"):
        import_zinc_diversity(
            output_root,
            zinc_root=zinc_root,
            zinc_manifest=manifest,
            target_per_ion_role=1,
        )

    assert before == {
        path.relative_to(output_root): path.read_bytes()
        for path in (output_root / "stage1/augmentation").rglob("*")
        if path.is_file()
    }


def test_resonance_eligibility_has_conservative_boundaries():
    assert resonance_eligibility("C" * 50, 50, 2).eligible
    heavy = resonance_eligibility("C" * 51, 50, 2)
    assert not heavy.eligible
    assert heavy.reason == "heavy_atom_limit"

    assert resonance_eligibility("[Fe+2]", 50, 2).eligible
    charge = resonance_eligibility("[Fe+3]", 50, 2)
    assert not charge.eligible
    assert charge.reason == "charge_limit"


def test_complex_multicharged_ion_skips_resonance_supplier(monkeypatch):
    complex_anion = (
        "O=C([O-])c1cc2c(C(=O)[O-])cc1Oc1nnc(c3c1CCC3)"
        "Oc1cc(C(=O)[O-])c(cc1C(=O)[O-])Oc1nnc(c3c1CCC3)"
        "Oc1cc(C(=O)[O-])c(cc1C(=O)[O-])Oc1nnc(c3c1CCC3)O2"
    )

    def forbidden_supplier(*_args, **_kwargs):
        raise AssertionError("ineligible molecule must not enumerate resonance")

    monkeypatch.setattr(Chem, "ResonanceMolSupplier", forbidden_supplier)
    generated, audit = training_splits._generate_rule_candidates_with_audit(
        complex_anion,
        "anion",
        resonance_max_structs=256,
        resonance_max_heavy_atoms=50,
        resonance_max_abs_charge=2,
    )

    assert audit["resonance_status"] == "skipped"
    assert set(audit["resonance_skip_reason"].split(";")) == {
        "charge_limit",
        "heavy_atom_limit",
    }
    assert all(row["rule"] != "resonance_equivalent" for row in generated)


def test_resonance_cap_is_reported_as_truncated():
    _generated, audit = training_splits._generate_rule_candidates_with_audit(
        "N#[N+][O-]",
        "molecule",
        resonance_max_structs=1,
        resonance_max_heavy_atoms=50,
        resonance_max_abs_charge=2,
    )

    assert audit["resonance_examined"] == 1
    assert audit["resonance_truncated"]


class PartiallyFailingPubChemClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def similarity(
        self,
        smiles: str,
        *,
        threshold: int,
        max_records: int,
    ) -> list[dict[str, object]]:
        canonical = canonicalize_smiles(smiles)
        self.calls.append(canonical)
        if canonical == "CCO":
            raise IncompletePubChemQuery("HTTP 500 for ethanol")
        assert canonical == "CCF"
        return [{"CID": 7, "SMILES": "CCCl", "Charge": 0}]

    def close(self) -> None:
        pass


class RecoveringPubChemClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def similarity(
        self,
        smiles: str,
        *,
        threshold: int,
        max_records: int,
    ) -> list[dict[str, object]]:
        canonical = canonicalize_smiles(smiles)
        self.calls.append(canonical)
        assert canonical == "CCO"
        return [{"CID": 8, "SMILES": "CO", "Charge": 0}]

    def close(self) -> None:
        pass


def test_partial_pubchem_failure_writes_output_and_resumes_only_failure(
    tmp_path: Path,
    monkeypatch,
):
    output_root = tmp_path / "training"
    write_stage1_molecules(output_root, ["CCO", "CCF"])
    first_client = PartiallyFailingPubChemClient()

    first = augment_pretraining_entities(
        output_root,
        client=first_client,
    )

    molecule_path = (
        output_root / "stage1" / "augmentation" / "molecule.csv"
    )
    first_molecule = pd.read_csv(molecule_path, keep_default_na=False)
    assert first["augmentation_entities"] == len(first_molecule)
    first_chloride = first_molecule.set_index("SMILES").loc["CCCl"]
    assert "pubchem" in first_chloride["origin_list"].split(";")
    assert str(first_chloride["pubchem_cid_list"]) == "7"
    audit_root = output_root / "stage1" / "augmentation" / "_audit"
    summary = json.loads(
        (audit_root / "augmentation_summary.json").read_text()
    )
    assert summary["completion_status"] == "partial_pubchem"
    failures = pd.read_csv(audit_root / "pubchem_failures.csv")
    assert failures["seed_smiles"].tolist() == ["CCO"]

    def forbidden_local_rules(*_args, **_kwargs):
        raise AssertionError("completed local rules must be resumed from cache")

    monkeypatch.setattr(
        training_splits,
        "_generate_rule_candidates_with_audit",
        forbidden_local_rules,
    )
    second_client = RecoveringPubChemClient()
    second = augment_pretraining_entities(
        output_root,
        client=second_client,
    )

    assert second_client.calls == ["CCO"]
    second_molecule = pd.read_csv(molecule_path, keep_default_na=False)
    assert second["augmentation_entities"] == len(second_molecule)
    assert set(first_molecule["SMILES"]) <= set(second_molecule["SMILES"])
    recovered = second_molecule.set_index("SMILES").loc["CO"]
    assert "pubchem" in recovered["origin_list"].split(";")
    assert str(recovered["pubchem_cid_list"]) == "8"
    summary = json.loads(
        (audit_root / "augmentation_summary.json").read_text()
    )
    assert summary["completion_status"] == "complete"
    assert pd.read_csv(audit_root / "pubchem_failures.csv").empty


def test_offline_missing_cache_fails_without_replacing_stage1(tmp_path: Path):
    output_root = tmp_path / "training"
    original = write_stage1_molecules(output_root, ["CCO"])
    previous_augmentation = write_existing_augmentation(output_root)

    with pytest.raises(IncompletePubChemQuery, match="offline"):
        augment_pretraining_entities(
            output_root,
            offline=True,
            client=PartiallyFailingPubChemClient(),
        )

    assert (output_root / "stage1" / "molecule.csv").read_bytes() == original
    assert previous_augmentation == {
        path: path.read_bytes() for path in previous_augmentation
    }


def test_interruption_preserves_stage1_and_local_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    output_root = tmp_path / "training"
    original = write_stage1_molecules(output_root, ["CCO"])
    previous_augmentation = write_existing_augmentation(output_root)

    class InterruptingClient:
        def similarity(self, *_args, **_kwargs):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        augment_pretraining_entities(
            output_root,
            client=InterruptingClient(),
        )
    assert (output_root / "stage1" / "molecule.csv").read_bytes() == original
    assert previous_augmentation == {
        path: path.read_bytes() for path in previous_augmentation
    }

    def forbidden_local_rules(*_args, **_kwargs):
        raise AssertionError("completed local rules must survive interruption")

    monkeypatch.setattr(
        training_splits,
        "_generate_rule_candidates_with_audit",
        forbidden_local_rules,
    )
    augment_pretraining_entities(
        output_root,
        client=EmptyPubChemClient(),
    )


def test_augmentation_parameter_change_invalidates_candidate_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    output_root = tmp_path / "training"
    write_stage1_molecules(output_root, ["CCO"])
    augment_pretraining_entities(
        output_root,
        resonance_max_structs=8,
        client=EmptyPubChemClient(),
    )

    original_generator = (
        training_splits._generate_rule_candidates_with_audit
    )
    calls = 0

    def counting_generator(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_generator(*args, **kwargs)

    monkeypatch.setattr(
        training_splits,
        "_generate_rule_candidates_with_audit",
        counting_generator,
    )
    augment_pretraining_entities(
        output_root,
        resonance_max_structs=9,
        client=EmptyPubChemClient(),
    )

    assert calls == 1


def test_augmentation_rejects_legacy_mixed_stage1_without_replacing_files(
    tmp_path: Path,
):
    output_root = tmp_path / "training"
    original = write_stage1_molecules(output_root, ["CCO"])
    stage1 = output_root / "stage1"
    molecule = pd.read_csv(stage1 / "molecule.csv", keep_default_na=False)
    molecule.loc[0, "origin_list"] = "dataset;pubchem"
    molecule.loc[0, "seed_smiles_list"] = "CCF"
    molecule.to_csv(stage1 / "molecule.csv", index=False)
    legacy = (stage1 / "molecule.csv").read_bytes()
    previous_augmentation = write_existing_augmentation(output_root)

    with pytest.raises(TrainingSplitError, match="extract-pretrain"):
        augment_pretraining_entities(
            output_root,
            client=EmptyPubChemClient(),
        )

    assert legacy != original
    assert (stage1 / "molecule.csv").read_bytes() == legacy
    assert previous_augmentation == {
        path: path.read_bytes() for path in previous_augmentation
    }


def test_augmentation_keeps_resonance_forms_as_independent_rows(
    tmp_path: Path,
):
    output_root = tmp_path / "training"
    stage1 = output_root / "stage1"
    stage1.mkdir(parents=True)
    pd.DataFrame(
        [{"cation": "CCCC[n+]1ccn(C)c1", "anion": "[Cl-]"}]
    ).to_csv(stage1 / "IL.csv", index=False)
    for role in ("anion", "molecule"):
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

    summary = augment_pretraining_entities(
        output_root,
        client=EmptyPubChemClient(),
    )

    augmented = pd.read_csv(
        stage1 / "augmentation" / "cation.csv",
        keep_default_na=False,
    )
    resonance = augmented[
        augmented["rule_list"].map(
            lambda value: "resonance_equivalent" in str(value)
        )
    ]
    assert summary["resonance_exported"] == 1
    assert len(resonance) == 1
    assert resonance.iloc[0]["origin_list"] == "rule"
    assert resonance.iloc[0]["seed_smiles_list"] == "CCCC[n+]1ccn(C)c1"
    assert fixed_h_identity_key(resonance.iloc[0]["SMILES"]) == (
        fixed_h_identity_key("CCCC[n+]1ccn(C)c1")
    )
    assert (stage1 / "IL.csv").read_bytes() == original_il


def test_neutral_rule_generation_shares_generic_ion_transformations():
    cases = {
        "CCCC": {"terminal_alkyl", "alkyl_branching"},
        "CCCl": {"halogen"},
        "CCO": {"hydroxyl_thiol"},
        "c1ccccc1": {"aromatic_CH_N"},
    }
    for seed, expected_families in cases.items():
        generated = generate_rule_candidates(seed, "molecule")
        families = {row["family"] for row in generated}
        assert expected_families <= families
        for row in generated:
            candidate = row["SMILES"]
            assert formal_charge(candidate) == 0
            assert fragment_count(candidate) == 1

    halogen_rules = {
        row["rule"] for row in generate_rule_candidates("CCCl", "molecule")
    }
    assert "halogen_Cl_to_I" in halogen_rules
    neutral_perfluoro = generate_rule_candidates(
        "FC(F)(F)C(F)(F)F",
        "molecule",
    )
    assert all(
        not row["rule"].startswith("perfluoroalkyl_")
        for row in neutral_perfluoro
    )
    assert all(
        not row["rule"].startswith("cation_headgroup_")
        for row in neutral_perfluoro
    )


def test_neutral_augmentation_combines_shared_rules_and_pubchem(
    tmp_path: Path,
):
    output_root = tmp_path / "training"
    stage1 = output_root / "stage1"
    stage1.mkdir(parents=True)
    pd.DataFrame(columns=["cation", "anion"]).to_csv(
        stage1 / "IL.csv",
        index=False,
    )
    for role in ("anion", "cation"):
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

    summary = augment_pretraining_entities(
        output_root,
        client=FakePubChemClient(),
    )

    assert summary["augmentation_entities"] > 2
    molecule = pd.read_csv(stage1 / "molecule.csv", keep_default_na=False)
    assert list(molecule.columns) == PRETRAIN_ENTITY_COLUMNS
    assert molecule.set_index("SMILES").loc["CCF", "origin_list"] == "dataset"
    augmented = pd.read_csv(
        stage1 / "augmentation" / "molecule.csv",
        keep_default_na=False,
    ).set_index("SMILES")
    assert {"CCBr", "CCO", "CCCl", "CCI"} <= set(augmented.index)
    pubchem_halogen = augmented.loc["CCBr"]
    assert pubchem_halogen["origin_list"] == "pubchem;rule"
    assert pubchem_halogen["seed_smiles_list"] == "CCF"
    assert pubchem_halogen["rule_list"] == "halogen_F_to_Br"
    assert str(pubchem_halogen["pubchem_cid_list"]) == "2"
    pubchem_only = augmented.loc["CCO"]
    assert pubchem_only["origin_list"] == "pubchem"
    assert str(pubchem_only["pubchem_cid_list"]) == "3"
    assert not (stage1 / "augmentation_provenance.csv").exists()


def test_neutral_ruleset_migration_preserves_completed_pubchem_state(
    tmp_path: Path,
    monkeypatch,
):
    output_root = tmp_path / "training"
    write_stage1_molecules(output_root, ["CCF"])
    augment_pretraining_entities(
        output_root,
        client=EmptyPubChemClient(),
    )
    cache = output_root / ".cache" / "stage1_augmentation.sqlite"
    with training_splits.sqlite3.connect(cache) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("neutral_ruleset_version", "0"),
        )

    original_generator = training_splits._generate_rule_candidates_with_audit
    generated_roles: list[str] = []

    def counting_generator(smiles, role, **kwargs):
        generated_roles.append(role)
        return original_generator(smiles, role, **kwargs)

    class ForbiddenPubChemClient:
        def similarity(self, *_args, **_kwargs):
            raise AssertionError("completed PubChem state must be preserved")

    monkeypatch.setattr(
        training_splits,
        "_generate_rule_candidates_with_audit",
        counting_generator,
    )
    augment_pretraining_entities(
        output_root,
        client=ForbiddenPubChemClient(),
    )

    assert generated_roles == ["molecule"]


def test_augmentation_exports_all_novel_candidates_and_consistent_audit(
    tmp_path: Path,
):
    output_root = tmp_path / "training"
    write_stage1_molecules(output_root, ["CCF", "CCO"])
    stage1 = output_root / "stage1"
    root_snapshot = {
        stage1 / f"{role}.csv": (stage1 / f"{role}.csv").read_bytes()
        for role in ("anion", "cation", "molecule")
    }

    summary = augment_pretraining_entities(
        output_root,
        client=AllCandidatePubChemClient(),
    )

    augmentation = stage1 / "augmentation"
    assert {
        path.name for path in augmentation.glob("*.csv")
    } == {"anion.csv", "cation.csv", "molecule.csv"}
    assert pd.read_csv(augmentation / "anion.csv").empty
    assert pd.read_csv(augmentation / "cation.csv").empty
    molecule = pd.read_csv(
        augmentation / "molecule.csv",
        keep_default_na=False,
    )
    assert {"CCBr", "CCC", "CN", "CO"} <= set(molecule["SMILES"])
    pubchem_rows = molecule[
        molecule["origin_list"].str.split(";").map(lambda origins: "pubchem" in origins)
    ]
    assert {"CCBr", "CCC", "CN", "CO"} <= set(pubchem_rows["SMILES"])
    assert summary == json.loads(
        (
            augmentation / "_audit" / "augmentation_summary.json"
        ).read_text()
    )
    assert summary["base_entities"] == 2
    assert summary["base_entities_by_role"] == {
        "anion": 0,
        "cation": 0,
        "molecule": 2,
    }
    assert summary["candidate_relation_rows"]["pubchem_similarity"] == 6
    assert summary["candidate_relation_rows"]["rule"] > 0
    assert summary["augmentation_entities"] == len(molecule)
    assert summary["augmentation_entities_by_role"] == {
        "anion": 0,
        "cation": 0,
        "molecule": len(molecule),
    }
    expected_origin_counts = {
        "rule_only": int((molecule["origin_list"] == "rule").sum()),
        "pubchem_only": int((molecule["origin_list"] == "pubchem").sum()),
        "both": int((molecule["origin_list"] == "pubchem;rule").sum()),
    }
    assert summary["augmentation_entities_by_origin"] == expected_origin_counts
    assert summary["excluded_base_overlaps"] >= 2
    assert root_snapshot == {
        path: path.read_bytes() for path in root_snapshot
    }


def test_neutral_resonance_overlap_merges_rule_and_pubchem_provenance(
    tmp_path: Path,
):
    output_root = tmp_path / "training"
    stage1 = output_root / "stage1"
    stage1.mkdir(parents=True)
    pd.DataFrame(columns=["cation", "anion"]).to_csv(
        stage1 / "IL.csv",
        index=False,
    )
    original_il = (stage1 / "IL.csv").read_bytes()
    for role in ("anion", "cation"):
        pd.DataFrame(columns=PRETRAIN_ENTITY_COLUMNS).to_csv(
            stage1 / f"{role}.csv",
            index=False,
        )
    pd.DataFrame(
        [
            {
                "SMILES": "N#[N+][O-]",
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

    summary = augment_pretraining_entities(
        output_root,
        client=NeutralResonancePubChemClient(),
    )

    molecule = pd.read_csv(
        stage1 / "augmentation" / "molecule.csv",
        keep_default_na=False,
    )
    resonance = molecule.set_index("SMILES").loc["[N-]=[N+]=O"]
    assert resonance["origin_list"] == "pubchem;rule"
    assert resonance["rule_list"] == "resonance_equivalent"
    assert resonance["seed_smiles_list"] == "N#[N+][O-]"
    assert str(resonance["pubchem_cid_list"]) == "2"
    assert summary["candidate_relation_rows"]["pubchem_similarity"] == 1
    assert summary["candidate_relation_rows"]["rule"] >= 1
    assert summary["augmentation_entities_by_origin"]["both"] == 1
    assert summary["resonance_generated"] == 1
    assert summary["resonance_exported"] == 1
    assert (stage1 / "IL.csv").read_bytes() == original_il


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


def _group_partition_signature(
    groups: pd.Series,
    folds: pd.Series,
) -> frozenset[frozenset[str]]:
    return frozenset(
        frozenset(groups.loc[folds.eq(fold)].astype(str))
        for fold in range(5)
    )


def test_task_group_kfold_is_disjoint_distinct_and_deterministic():
    groups = pd.Series(
        [
            group
            for index in range(24)
            for group in [f"group_{index}"] * (index % 5 + 1)
        ]
    )
    first = task_group_kfold_assignments(
        groups,
        task_id="experiment/task_a",
        strategy="cation",
        repeats=5,
        seed=42,
    )
    repeated = task_group_kfold_assignments(
        groups,
        task_id="experiment/task_a",
        strategy="cation",
        repeats=5,
        seed=42,
    )

    assert len(first) == 5
    assert [folds.tolist() for folds in first] == [
        folds.tolist() for folds in repeated
    ]
    signatures = {
        _group_partition_signature(groups, folds)
        for folds in first
    }
    assert len(signatures) == 5
    for folds in first:
        assert set(folds) == set(range(5))
        assignments = pd.DataFrame(
            {"group": groups, "fold": folds}
        ).drop_duplicates()
        assert assignments.groupby("group")["fold"].nunique().eq(1).all()

    other_task = task_group_kfold_assignments(
        groups,
        task_id="experiment/task_b",
        strategy="cation",
        repeats=5,
        seed=42,
    )
    assert {
        _group_partition_signature(groups, folds)
        for folds in other_task
    } != signatures


def test_task_group_kfold_rejects_too_few_or_indistinguishable_groups():
    with pytest.raises(TrainingSplitError, match="Fewer than five"):
        task_group_kfold_assignments(
            pd.Series(["a", "b", "c", "d"]),
            task_id="experiment/task",
            strategy="anion",
            repeats=1,
            seed=42,
        )

    with pytest.raises(
        TrainingSplitError,
        match="Unable to build 2 distinct",
    ):
        task_group_kfold_assignments(
            pd.Series(["a", "b", "c", "d", "e"]),
            task_id="experiment/task",
            strategy="anion",
            repeats=2,
            seed=42,
            max_candidates=8,
        )


def test_partial_charge_rows_keep_mol_ids_and_validate_formal_charge(tmp_path: Path):
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
            {
                "mol_id": "mol_3",
                "SMILES": "[NH4+]",
                "charge": 1,
                "source_list": "simulation",
            },
            {
                "mol_id": "mol_4",
                "SMILES": "[Cl-]",
                "charge": -1,
                "source_list": "simulation",
            },
        ],
    )
    task = TaskSpec(
        task_id="simulation/partial_atomic_charge",
        stage=2,
        source_file="simulation/charge.csv",
        target_columns=("partial_atomic_charge",),
        identity_columns=("SMILES",),
        system_type="molecule",
    )

    frame, raw_rows = prepare_task_frame(final_root, task, "checksum")
    assert raw_rows == 4
    assert len(frame) == 4
    assert frame["SMILES"].tolist()[:2] == ["CCO", "CCO"]
    assert frame["mol_id"].tolist() == ["mol_1", "mol_2", "mol_3", "mol_4"]
    assert frame["formal_charge"].tolist() == [0, 0, 1, -1]
    assert frame["role"].tolist() == ["neutral", "neutral", "cation", "anion"]
    assert frame.iloc[:2]["_system_id"].nunique() == 1

    conflicting = pd.read_csv(path)
    conflicting.loc[1, "charge"] = -1
    conflicting.to_csv(path, index=False)
    with pytest.raises(TrainingSplitError, match="Formal charge mismatch"):
        prepare_task_frame(final_root, task, "checksum")

    duplicated = pd.read_csv(path)
    duplicated.loc[1, "charge"] = 0
    duplicated.loc[1, "mol_id"] = "mol_1"
    duplicated.to_csv(path, index=False)
    with pytest.raises(TrainingSplitError, match="Duplicate mol_id"):
        prepare_task_frame(final_root, task, "checksum")


def test_partial_charge_missing_structure_is_excluded_and_audited(tmp_path: Path):
    final_root = tmp_path / "final"
    write_csv(
        final_root / "simulation" / "charge.csv",
        [
            {
                "mol_id": "mol_present",
                "SMILES": "CCO",
                "charge": 0,
                "source_list": "simulation",
            },
            {
                "mol_id": "mol_missing",
                "SMILES": "CCN",
                "charge": 0,
                "source_list": "simulation",
            },
        ],
    )
    manifest_relative = "simulation/charge_20260514/structure_manifest.csv"
    write_csv(
        final_root / manifest_relative,
        [
            {
                "mol_id": "mol_present",
                "relative_path": "mol_present.mol2",
                "format": "mol2",
                "size_bytes": 1,
                "sha256": "0" * 64,
            }
        ],
    )
    task = TaskSpec(
        "simulation/partial_atomic_charge",
        2,
        "simulation/charge.csv",
        ("partial_atomic_charge",),
        ("SMILES",),
        "molecule",
        resource_manifest=manifest_relative,
    )
    frame, _ = prepare_task_frame(final_root, task, "checksum")

    filtered, audit = training_splits.exclude_missing_partial_charge_resources(
        frame,
        task,
        final_root,
    )

    assert filtered["mol_id"].tolist() == ["mol_present"]
    assert audit.to_dict("records") == [
        {
            "task_id": "simulation/partial_atomic_charge",
            "mol_id": "mol_missing",
            "SMILES": "CCN",
            "reason": "missing_structure_resource",
        }
    ]


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

    large_pairs = [ion_pair(index) for index in range(51, 552)]
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

    write_csv(
        final_root / "experiment" / "heat_capacity.csv",
        [
            {
                "cation": ion_pair(index)[0],
                "anion": ion_pair(index)[1],
                "temperature_K": 298.15,
                "pressure_kPa": 101.325,
                "heat_capacity_J/mol/K": float(index),
                "source_list": "experiment",
            }
            for index in [2, *range(200, 209)]
        ],
    )
    write_csv(
        final_root
        / "experiment"
        / "isobaric_coefficient_of_volume_expansion.csv",
        [
            {
                "cation": ion_pair(index)[0],
                "anion": ion_pair(index)[1],
                "temperature_K": 298.15,
                "pressure_kPa": 101.325,
                "isobaric_coefficient_of_volume_expansion_K^-1": float(index),
                "source_list": "experiment",
            }
            for index in [3, *range(300, 309)]
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
    ignored_structure.parent.mkdir(parents=True, exist_ok=True)
    ignored_structure.write_text("invalid structure", encoding="utf-8")

    output_root = tmp_path / "training"
    extract_pretraining_entities(final_root, output_root)
    catalog = build_training_splits(final_root, output_root, seed=42)

    assert int(catalog["stage"].eq(2).sum()) == 9
    assert set(
        catalog.loc[catalog["stage"].eq(2), "task_id"]
    ) == {
        "simulation/pbe_tzvp_cation_orbitals",
        "simulation/pbe_tzvp_anion_orbitals",
        "simulation/partial_atomic_charge",
        "simulation/density",
        "simulation/heat_capacity",
        "simulation/heat_of_vaporization",
        "simulation/simulated_qm_elec_hf",
        "simulation/thermal_expansion",
        "simulation/transfer_organic",
    }
    stage2 = catalog[catalog["stage"].eq(2)].set_index("task_id")
    assert not set(stage2.index) & set(
        catalog.loc[catalog["stage"].eq(3), "task_id"]
    )
    assert catalog.loc[catalog["stage"].eq(3), "task_id"].str.startswith(
        "experiment/"
    ).all()
    assert stage2.loc[
        [
            "simulation/density",
            "simulation/heat_capacity",
            "simulation/thermal_expansion",
            "simulation/transfer_organic",
        ],
        ["raw_rows", "rows", "unique_systems"],
    ].to_dict("index") == {
        "simulation/density": {
            "raw_rows": 101,
            "rows": 51,
            "unique_systems": 50,
        },
        "simulation/heat_capacity": {
            "raw_rows": 101,
            "rows": 100,
            "unique_systems": 99,
        },
        "simulation/thermal_expansion": {
            "raw_rows": 101,
            "rows": 100,
            "unique_systems": 99,
        },
        "simulation/transfer_organic": {
            "raw_rows": 101,
            "rows": 50,
            "unique_systems": 50,
        },
    }
    stage3 = catalog[catalog["stage"].eq(3)].set_index("task_id")
    assert stage3.loc["experiment/density", "tier"] == "large"
    assert stage3.loc["experiment/solvation", "tier"] == "medium"
    assert stage3.loc[
        "experiment/transfer_organic",
        ["raw_rows", "rows", "unique_systems"],
    ].tolist() == [50, 50, 50]
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

    def il_fold_assignment(task_root: Path) -> dict[tuple[str, str], int]:
        return {
            pair: fold
            for fold in range(1, 6)
            for pair in pd.read_csv(
                task_root / "IL" / f"fold{fold}.csv"
            )[["cation", "anion"]].itertuples(index=False, name=None)
        }

    assert il_fold_assignment(density_root) != il_fold_assignment(
        conductivity_root
    )

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
    for strategy in ("IL", "cation", "anion"):
        signatures = set()
        for repeat in range(1, 6):
            fold_groups = []
            for fold in range(1, 6):
                frame = pd.read_csv(
                    small_root
                    / strategy
                    / f"cv{repeat}"
                    / f"fold{fold}.csv"
                )
                columns = {
                    "IL": ["cation", "anion"],
                    "cation": ["cation"],
                    "anion": ["anion"],
                }[strategy]
                fold_groups.append(
                    frozenset(
                        frame[columns]
                        .astype(str)
                        .agg("\x1f".join, axis=1)
                    )
                )
            signatures.add(frozenset(fold_groups))
        assert len(signatures) == 5

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
    assert len(train) + len(valid) == 51

    def stage2_assignments(
        task_name: str,
        columns: list[str],
    ) -> dict[tuple[str, ...], str]:
        assignments: dict[tuple[str, ...], str] = {}
        task_root = output_root / "stage2" / task_name
        for partition, filename in (
            ("train", "train.csv"),
            ("valid", "valid.csv"),
        ):
            frame = pd.read_csv(task_root / filename)
            for system in frame[columns].itertuples(index=False, name=None):
                previous = assignments.setdefault(system, partition)
                assert previous == partition
        return assignments

    density_assignments = stage2_assignments(
        "density",
        ["cation", "anion"],
    )
    heat_capacity_assignments = stage2_assignments(
        "heat_capacity",
        ["cation", "anion"],
    )
    thermal_expansion_assignments = stage2_assignments(
        "thermal_expansion",
        ["cation", "anion"],
    )
    stage2_assignments("simulated_qm_elec_hf", ["SMILES"])
    stage2_assignments("pbe_tzvp_cation_orbitals", ["cation"])
    stage2_assignments("pbe_tzvp_anion_orbitals", ["anion"])
    stage2_assignments("partial_atomic_charge", ["SMILES"])
    heat_of_vaporization_assignments = stage2_assignments(
        "heat_of_vaporization",
        ["cation", "anion"],
    )
    transfer_assignments = stage2_assignments(
        "transfer_organic",
        ["solute", "solvent"],
    )
    assert density_assignments != heat_capacity_assignments

    assert set(density_assignments) == {
        canonical_ion_pair(index) for index in range(1, 51)
    }
    assert canonical_ion_pair(2) not in heat_capacity_assignments
    assert canonical_ion_pair(51) in heat_capacity_assignments
    assert canonical_ion_pair(3) not in thermal_expansion_assignments
    assert canonical_ion_pair(2) in thermal_expansion_assignments
    assert canonical_ion_pair(51) in thermal_expansion_assignments
    assert set(transfer_assignments) == {
        canonical_transfer_system(index) for index in range(51, 101)
    }
    assert set(heat_of_vaporization_assignments) == {
        canonical_ion_pair(index) for index in range(1, 101)
    }

    transfer_stage3_root = (
        output_root / "stage3" / "experiment" / "transfer_organic"
    )
    transfer_stage3_folds = [
        pd.read_csv(
            transfer_stage3_root / "random" / "cv1" / f"fold{fold}.csv"
        )
        for fold in range(1, 6)
    ]
    assert sum(map(len, transfer_stage3_folds)) == 50

    overlap_audit = pd.read_csv(
        output_root / "_audit" / "stage2_overlap_exclusions.csv"
    )
    assert list(overlap_audit.columns) == [
        "stage2_task_id",
        "stage3_task_id",
        "cation",
        "anion",
        "excluded_stage2_rows",
        "matching_stage3_rows",
        "solute",
        "solvent",
    ]
    assert overlap_audit.groupby("stage2_task_id").size().to_dict() == {
        "simulation/density": 50,
        "simulation/heat_capacity": 1,
        "simulation/thermal_expansion": 1,
        "simulation/transfer_organic": 50,
    }
    assert overlap_audit.groupby("stage2_task_id")[
        "excluded_stage2_rows"
    ].sum().to_dict() == {
        "simulation/density": 50,
        "simulation/heat_capacity": 1,
        "simulation/thermal_expansion": 1,
        "simulation/transfer_organic": 51,
    }
    assert overlap_audit["matching_stage3_rows"].eq(1).all()
    density_overlap = overlap_audit.loc[
        overlap_audit["stage2_task_id"].eq("simulation/density")
    ]
    assert set(
        density_overlap[["cation", "anion"]].itertuples(
            index=False,
            name=None,
        )
    ) == {canonical_ion_pair(index) for index in range(51, 101)}
    assert density_overlap["excluded_stage2_rows"].eq(1).all()
    assert overlap_audit.loc[
        overlap_audit["stage2_task_id"].eq("simulation/heat_capacity"),
        "stage3_task_id",
    ].tolist() == ["experiment/heat_capacity"]
    assert overlap_audit.loc[
        overlap_audit["stage2_task_id"].eq(
            "simulation/thermal_expansion"
        ),
        "stage3_task_id",
    ].tolist() == [
        "experiment/isobaric_coefficient_of_volume_expansion"
    ]
    il_overlap = overlap_audit.loc[
        overlap_audit["stage2_task_id"].ne(
            "simulation/transfer_organic"
        )
    ]
    assert il_overlap[["solute", "solvent"]].isna().all().all()
    transfer_overlap = overlap_audit.loc[
        overlap_audit["stage2_task_id"].eq(
            "simulation/transfer_organic"
        )
    ]
    assert transfer_overlap[["cation", "anion"]].isna().all().all()
    assert set(
        transfer_overlap[["solute", "solvent"]].itertuples(
            index=False,
            name=None,
        )
    ) == {canonical_transfer_system(index) for index in range(1, 51)}
    assert transfer_overlap["stage3_task_id"].eq(
        "experiment/transfer_organic"
    ).all()
    assert transfer_overlap["matching_stage3_rows"].eq(1).all()

    repeated_rows = pd.concat([train, valid], ignore_index=True)
    repeated_pair = repeated_rows.loc[
        repeated_rows["temperature_K"].eq(320.0),
        ["cation", "anion"],
    ].iloc[0]
    repeated_rows = repeated_rows.loc[
        repeated_rows["cation"].eq(repeated_pair["cation"])
        & repeated_rows["anion"].eq(repeated_pair["anion"])
    ]
    assert set(repeated_rows["temperature_K"]) == {298.0, 320.0}
    assert len(repeated_rows) == 2

    legacy_paths = [
        output_root / "manifest.json",
        output_root / "audit",
        output_root / "stage2" / "rows.csv",
        output_root / "stage3" / "rows.csv",
        output_root / "stage3" / "test_groups.csv",
    ]
    assert not any(path.exists() for path in legacy_paths)
    written_catalog = pd.read_csv(
        output_root / "task_catalog.csv",
        keep_default_na=False,
    )
    pd.testing.assert_frame_equal(written_catalog, catalog, check_dtype=False)
    required_catalog_columns = {
        "catalog_schema_version",
        "task_kind",
        "target_level",
        "condition_columns",
        "split_unit",
        "sample_unit",
        "simulation_method",
        "experiment_reference",
        "materialized_path",
        "label_source",
        "resource_manifest",
    }
    assert required_catalog_columns <= set(written_catalog.columns)
    partial_catalog = stage2.loc["simulation/partial_atomic_charge"]
    assert partial_catalog["target_columns"] == "partial_atomic_charge"
    assert partial_catalog["task_kind"] == "atom_property"
    assert partial_catalog["target_level"] == "atom"
    assert partial_catalog["sample_unit"] == "mol_id"
    assert partial_catalog["split_unit"] == "SMILES"
    assert partial_catalog["label_source"] == "structure_resource"
    assert partial_catalog["materialized_path"] == "stage2/partial_atomic_charge"
    partial_rows = pd.concat(
        [
            pd.read_csv(output_root / "stage2" / "partial_atomic_charge" / name)
            for name in ("train.csv", "valid.csv")
        ],
        ignore_index=True,
    )
    assert list(partial_rows.columns) == [
        "mol_id",
        "SMILES",
        "role",
        "formal_charge",
        "source_list",
    ]
    assert len(partial_rows) == 101
    assert partial_rows["mol_id"].nunique() == 101
    partitions = {}
    for partition in ("train", "valid"):
        for mol_id in pd.read_csv(
            output_root / "stage2" / "partial_atomic_charge" / f"{partition}.csv"
        )["mol_id"]:
            partitions[mol_id] = partition
    assert partitions["mol_0000001"] == partitions["mol_duplicate"]
    partial_resource_audit = pd.read_csv(
        output_root / "_audit" / "partial_atomic_charge_resource_exclusions.csv"
    )
    assert list(partial_resource_audit.columns) == [
        "task_id",
        "mol_id",
        "SMILES",
        "reason",
    ]
    assert partial_resource_audit.empty
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

    build_training_splits(final_root, output_root, seed=43)
    assert density_assignments != stage2_assignments(
        "density",
        ["cation", "anion"],
    )


def test_discover_tasks_requires_stage2_overlap_references(tmp_path: Path):
    """Require every experiment file used for Stage-2 overlap filtering."""
    final_root = tmp_path / "final"
    write_stage2_files(final_root)
    cation, anion = ion_pair(1)
    write_csv(
        final_root / "experiment" / "density.csv",
        [
            {
                "cation": cation,
                "anion": anion,
                "temperature_K": 298.15,
                "density_g/cm^3": 1.0,
                "source_list": "experiment",
            }
        ],
    )
    write_csv(
        final_root / "experiment" / "heat_capacity.csv",
        [
            {
                "cation": cation,
                "anion": anion,
                "temperature_K": 298.15,
                "heat_capacity_J/mol/K": 1.0,
                "source_list": "experiment",
            }
        ],
    )
    write_csv(
        final_root
        / "experiment"
        / "isobaric_coefficient_of_volume_expansion.csv",
        [
            {
                "cation": cation,
                "anion": anion,
                "temperature_K": 298.15,
                "isobaric_coefficient_of_volume_expansion_K^-1": 1.0,
                "source_list": "experiment",
            }
        ],
    )

    with pytest.raises(
        training_splits.TrainingSplitError,
        match=(
            "Missing required stage-3 overlap references: "
            "experiment/transfer_organic.csv"
        ),
    ):
        discover_tasks(final_root)


def test_discover_tasks_rejects_unclassified_simulation_dataset(tmp_path: Path):
    final_root = tmp_path / "final"
    write_stage2_files(final_root)
    cation, anion = ion_pair(1)
    for filename, target in (
        ("density.csv", "density_g/cm^3"),
        ("heat_capacity.csv", "heat_capacity_J/mol/K"),
        (
            "isobaric_coefficient_of_volume_expansion.csv",
            "isobaric_coefficient_of_volume_expansion_K^-1",
        ),
    ):
        write_csv(
            final_root / "experiment" / filename,
            [
                {
                    "cation": cation,
                    "anion": anion,
                    target: 1.0,
                    "source_list": "experiment",
                }
            ],
        )
    write_csv(
        final_root / "experiment" / "transfer_organic.csv",
        [
            {
                "solute": "CC",
                "solvent": "CO",
                "transfer_organic_kcal/mol": 1.0,
                "source_list": "experiment",
            }
        ],
    )
    write_csv(
        final_root / "simulation" / "unknown_property.csv",
        [
            {
                "SMILES": "CC",
                "unknown": 1.0,
                "source_list": "simulation",
            }
        ],
    )

    with pytest.raises(
        TrainingSplitError,
        match="Unclassified simulation datasets: simulation/unknown_property.csv",
    ):
        discover_tasks(final_root)


def test_overlap_exclusion_rejects_mismatched_system_types():
    """Reject overlap references that do not share one system identity."""
    task = TaskSpec(
        "simulation/transfer_organic",
        2,
        "simulation/transfer_organic.csv",
        ("transfer_organic_kcal/mol",),
        ("solute", "solvent"),
        "solute_solvent",
    )
    reference_task = TaskSpec(
        "experiment/density",
        3,
        "experiment/density.csv",
        ("density_g/cm^3",),
        ("cation", "anion"),
        "il",
    )

    with pytest.raises(
        training_splits.TrainingSplitError,
        match="matching supported system type",
    ):
        training_splits.exclude_stage2_experiment_overlap(
            pd.DataFrame(),
            task,
            reference_task,
            pd.DataFrame(),
        )
