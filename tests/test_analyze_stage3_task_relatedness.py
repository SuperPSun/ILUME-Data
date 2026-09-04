import csv
import gzip
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from scripts.analyze_stage3_task_relatedness import (
    CHEMISTRY_WEIGHT,
    CONDITION_WEIGHT,
    DEFAULT_EXPERIMENT_PRESSURE_KPA,
    DEFAULT_REFRACTIVE_INDEX_WAVELENGTH_NM,
    EXPECTED_STAGE3_TASK_IDS,
    _PairProgress,
    _load_task,
    _spearman,
    analyze_stage3_task_relatedness,
    brute_force_match_task_pair,
    chemistry_similarities,
    compute_condition_scales,
    condition_similarity,
    match_task_pair,
)


class TtyBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def load_fixture_task(
    tmp_path: Path,
    task_id: str,
    rows: list[dict[str, object]],
):
    path = tmp_path / f"{task_id.replace('/', '_')}.csv"
    write_csv(path, rows)
    return _load_task(
        task_id,
        path,
        {},
        rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048),
    )


def il_row(
    cation: str,
    anion: str,
    value: float,
    **conditions: object,
) -> dict[str, object]:
    return {"cation": cation, "anion": anion, **conditions, "property": value}


def test_pair_progress_shows_current_pair_completion_and_elapsed_time():
    stream = TtyBuffer()
    progress = _PairProgress(total_pairs=2, total_work=3, stream=stream)

    progress.start_pair(
        "experiment/density", "experiment/viscosity", pair_work=2
    )
    progress.advance_work()
    progress.advance_work()
    progress.finish_pair()
    progress.start_pair(
        "experiment/density", "experiment/solvation", pair_work=1
    )
    progress.advance_work()
    progress.finish_pair()
    progress.close()

    output = stream.getvalue()
    assert "0/3" in output
    assert "density -> viscosity" in output
    assert "3/3" in output
    assert "pair   2/2" in output
    assert "rows 1/1" in output
    assert "elapsed" in output
    assert "current: done" in output
    assert output.endswith("\n")


def test_pair_progress_is_silent_for_non_tty_streams():
    stream = io.StringIO()
    progress = _PairProgress(total_pairs=1, total_work=1, stream=stream)

    progress.start_pair(
        "experiment/density", "experiment/viscosity", pair_work=1
    )
    progress.advance_work()
    progress.finish_pair()
    progress.close()

    assert stream.getvalue() == ""


def test_chemistry_similarity_is_geometric_mean_by_role():
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    query = tuple(
        generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        for smiles in ("C[N+](C)(C)C", "[Cl-]")
    )
    target = tuple(
        generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        for smiles in ("CC[N+](C)(C)C", "[Br-]")
    )
    component_scores = [
        DataStructs.TanimotoSimilarity(query[index], target[index])
        for index in range(2)
    ]

    actual = chemistry_similarities(query, [target])[0]

    assert actual == pytest.approx(math.sqrt(np.prod(component_scores)))


def test_condition_similarity_uses_defaults_log_scales_and_neutral_missing(
    tmp_path: Path,
):
    source = load_fixture_task(
        tmp_path,
        "experiment/density",
        [
            {
                "cation": "C[N+](C)(C)C",
                "anion": "[Cl-]",
                "temperature_K": 300.0,
                "property": 1.0,
            }
        ],
    )
    target = load_fixture_task(
        tmp_path,
        "experiment/refractive_index",
        [
            {
                "cation": "C[N+](C)(C)C",
                "anion": "[Cl-]",
                "temperature_K": np.nan,
                "property": 1.4,
            }
        ],
    )
    scales = compute_condition_scales([source, target])

    assert source.frame.loc[0, "pressure_kPa"] == DEFAULT_EXPERIMENT_PRESSURE_KPA
    assert target.frame.loc[0, "pressure_kPa"] == DEFAULT_EXPERIMENT_PRESSURE_KPA
    assert (
        target.frame.loc[0, "wavelength_nm"]
        == DEFAULT_REFRACTIVE_INDEX_WAVELENGTH_NM
    )
    # Pressure matches; temperature is missing on one side; wavelength is absent
    # from density and therefore neutral: mean(1.0, 0.5, 0.5).
    actual = condition_similarity(source, 0, target, np.asarray([0]), scales)[0]
    assert actual == pytest.approx(2.0 / 3.0)


def test_no_applicable_conditions_are_neutral(tmp_path: Path):
    source = load_fixture_task(
        tmp_path,
        "experiment/source",
        [il_row("C[N+](C)(C)C", "[Cl-]", 1.0)],
    )
    target = load_fixture_task(
        tmp_path,
        "experiment/target",
        [il_row("CC[N+](C)(C)C", "[Br-]", 2.0)],
    )

    actual = condition_similarity(
        source, 0, target, np.asarray([0]), compute_condition_scales([source, target])
    )[0]

    assert actual == 0.5


def test_pruned_matcher_equals_brute_force_with_noncontiguous_groups(
    tmp_path: Path,
):
    source = load_fixture_task(
        tmp_path,
        "experiment/source",
        [
            il_row("C[N+](C)(C)C", "[Cl-]", 1.0, temperature_K=290.0),
            il_row("CC[N+](C)(C)C", "[Br-]", 2.0, temperature_K=310.0),
            il_row("C[N+](C)(C)C", "[Cl-]", 3.0, temperature_K=330.0),
        ],
    )
    target = load_fixture_task(
        tmp_path,
        "experiment/target",
        [
            il_row("CC[N+](C)(C)C", "[Br-]", 9.0, temperature_K=315.0),
            il_row("C[N+](C)(C)C", "[Cl-]", 8.0, temperature_K=335.0),
            il_row("CC[N+](C)(C)C", "[Br-]", 7.0, temperature_K=295.0),
            il_row("CCC[N+](C)(C)C", "[I-]", 6.0, temperature_K=300.0),
        ],
    )
    scales = compute_condition_scales([source, target])

    pruned = match_task_pair(source, target, scales)
    brute_force = brute_force_match_task_pair(source, target, scales)

    np.testing.assert_array_equal(pruned.target_indices, brute_force.target_indices)
    np.testing.assert_allclose(
        pruned.chemistry_similarity, brute_force.chemistry_similarity
    )
    np.testing.assert_allclose(
        pruned.condition_similarity, brute_force.condition_similarity
    )
    np.testing.assert_allclose(pruned.combined_similarity, brute_force.combined_similarity)


@pytest.mark.parametrize("seed", [3, 17, 41])
def test_pruned_matcher_equals_brute_force_on_random_small_inputs(
    tmp_path: Path, seed: int
):
    rng = np.random.default_rng(seed)
    cations = ["C[N+](C)(C)C", "CC[N+](C)(C)C", "CCC[N+](C)(C)C"]
    anions = ["[Cl-]", "[Br-]", "[I-]"]

    def rows(count: int) -> list[dict[str, object]]:
        return [
            il_row(
                cations[int(rng.integers(len(cations)))],
                anions[int(rng.integers(len(anions)))],
                float(index),
                temperature_K=float(rng.choice([290.0, 310.0, 330.0])),
                phase=str(rng.choice(["Liquid", "Solid"])),
            )
            for index in range(count)
        ]

    source = load_fixture_task(tmp_path, f"experiment/source_{seed}", rows(8))
    target = load_fixture_task(tmp_path, f"experiment/target_{seed}", rows(7))
    scales = compute_condition_scales([source, target])

    pruned = match_task_pair(source, target, scales)
    brute_force = brute_force_match_task_pair(source, target, scales)

    np.testing.assert_array_equal(pruned.target_indices, brute_force.target_indices)
    np.testing.assert_allclose(pruned.combined_similarity, brute_force.combined_similarity)


def test_duplicate_target_ties_select_earliest_row(tmp_path: Path):
    duplicate_rows = [
        il_row("C[N+](C)(C)C", "[Cl-]", 1.0),
        il_row("C[N+](C)(C)C", "[Cl-]", 2.0),
    ]
    source = load_fixture_task(tmp_path, "experiment/source", duplicate_rows)
    target = load_fixture_task(tmp_path, "experiment/target", duplicate_rows)
    scales = compute_condition_scales([source, target])

    cross_task = match_task_pair(source, target, scales)

    np.testing.assert_array_equal(cross_task.target_indices, [0, 0])


def test_low_similarity_query_still_gets_one_match(tmp_path: Path):
    source = load_fixture_task(
        tmp_path,
        "experiment/source",
        [il_row("[Na+]", "[Cl-]", 1.0)],
    )
    target = load_fixture_task(
        tmp_path,
        "experiment/target",
        [il_row("C[N+](C)(C)C", "[B-](F)(F)(F)F", 2.0)],
    )

    matches = match_task_pair(
        source, target, compute_condition_scales([source, target])
    )

    np.testing.assert_array_equal(matches.target_indices, [0])
    assert matches.combined_similarity[0] == pytest.approx(
        CHEMISTRY_WEIGHT * matches.chemistry_similarity[0]
        + CONDITION_WEIGHT * 0.5
    )


def test_spearman_keeps_sign_accepts_two_pairs_and_reports_constants():
    rho, reason = _spearman(np.asarray([1.0, 2.0]), np.asarray([9.0, 3.0]))
    assert rho == pytest.approx(-1.0)
    assert reason == ""

    rho, reason = _spearman(np.asarray([1.0, 1.0]), np.asarray([3.0, 4.0]))
    assert np.isnan(rho)
    assert reason == "constant_source_property"


def test_directed_nearest_neighbors_can_produce_asymmetric_correlations(
    tmp_path: Path,
):
    chemistry = {"cation": "C[N+](C)(C)C", "anion": "[Cl-]"}
    task_a = load_fixture_task(
        tmp_path,
        "experiment/task_a",
        [
            {**chemistry, "temperature_K": 290.0, "property": 1.0},
            {**chemistry, "temperature_K": 310.0, "property": 2.0},
            {**chemistry, "temperature_K": 330.0, "property": 3.0},
        ],
    )
    task_b = load_fixture_task(
        tmp_path,
        "experiment/task_b",
        [
            {**chemistry, "temperature_K": 290.0, "property": 1.0},
            {**chemistry, "temperature_K": 330.0, "property": 0.0},
        ],
    )
    scales = compute_condition_scales([task_a, task_b])

    a_to_b = match_task_pair(task_a, task_b, scales)
    b_to_a = match_task_pair(task_b, task_a, scales)
    rho_ab, _ = _spearman(
        task_a.property_values, task_b.property_values[a_to_b.target_indices]
    )
    rho_ba, _ = _spearman(
        task_b.property_values, task_a.property_values[b_to_a.target_indices]
    )

    assert rho_ab < 0.0
    assert rho_ba == pytest.approx(-1.0)
    assert rho_ab != pytest.approx(rho_ba)


def _topology_for_task(task_id: str) -> tuple[str, ...]:
    if task_id in {"experiment/solvation", "experiment/transfer"}:
        return ("cation", "anion", "solute")
    if task_id == "experiment/transfer_organic":
        return ("solute", "solvent")
    return ("cation", "anion")


def test_full_analysis_writes_directed_21_by_21_outputs(tmp_path: Path):
    input_root = tmp_path / "final"
    for task_index, task_id in enumerate(EXPECTED_STAGE3_TASK_IDS):
        roles = _topology_for_task(task_id)
        first = {
            "cation": "C[N+](C)(C)C",
            "anion": "[Cl-]",
            "solute": "CCO",
            "solvent": "CC",
        }
        second = {
            "cation": "CC[N+](C)(C)C",
            "anion": "[Br-]",
            "solute": "CCN",
            "solvent": "CCC",
        }
        rows = [
            {
                **{role: first[role] for role in roles},
                "temperature_K": 300.0,
                f"property_{task_index}": 1.0,
            },
            {
                **{role: second[role] for role in roles},
                "temperature_K": 320.0,
                f"property_{task_index}": 2.0,
            },
        ]
        write_csv(input_root / f"{task_id}.csv", rows)
    output_dir = tmp_path / "analysis"

    matrix, summary = analyze_stage3_task_relatedness(input_root, output_dir)

    assert matrix.shape == (21, 21)
    assert tuple(matrix.index) == EXPECTED_STAGE3_TASK_IDS
    assert tuple(matrix.columns) == EXPECTED_STAGE3_TASK_IDS
    assert np.isnan(np.diag(matrix)).all()
    assert np.isnan(
        matrix.loc["experiment/density", "experiment/transfer_organic"]
    )
    assert len(summary) == 21 * 21
    assert int(summary["compatible"].sum()) == 18**2 + 2**2 + 1
    assert int(summary["computed"].sum()) == 18 * 17 + 2 * 1
    assert summary["excluded_query_count"].eq(0).all()
    diagonal = summary[summary["source_task"].eq(summary["target_task"])]
    assert len(diagonal) == 21
    assert diagonal["computed"].eq(False).all()
    assert diagonal["matched_pairs"].eq(0).all()
    assert diagonal["undefined_reason"].eq("same_task_skipped").all()

    written_matrix = pd.read_csv(output_dir / "spearman_matrix.csv", index_col=0)
    assert written_matrix.shape == (21, 21)
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["similarity"]["threshold"] is None
    assert metadata["task_order"] == list(EXPECTED_STAGE3_TASK_IDS)
    with gzip.open(output_dir / "matches.csv.gz", "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 18 * 17 * 2 + 2 * 1 * 2
    assert set(rows[0]) >= {
        "source_task",
        "target_task",
        "chemistry_similarity",
        "condition_similarity",
        "combined_similarity",
    }
