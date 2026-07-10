from pathlib import Path

from scripts.build_final_data import build_final_data


def test_build_final_data_copies_buckets_and_excludes_requested_experiment_properties(tmp_path: Path):
    merged_root = tmp_path / "merged"
    experiment = merged_root / "experiment"
    simulation = merged_root / "simulation"
    experiment.mkdir(parents=True)
    simulation.mkdir()

    retained_experiment = experiment / "density.csv"
    retained_experiment.write_bytes(b"density,1\\n")
    for filename in (
        "thermal_diffusivity.csv",
        "enthalpy.csv",
        "entropy.csv",
        "heat_capacity_at_vapor_saturation_pressure.csv",
        "enthalpy_of_vaporization_or_sublimation.csv",
        "enthalpy_of_transition_or_fusion.csv",
    ):
        (experiment / filename).write_bytes(b"excluded,1\\n")
    retained_simulation = simulation / "heat_of_vaporization.csv"
    retained_simulation.write_bytes(b"heat_of_vaporization,1\\n")

    final_root = tmp_path / "final"
    stale_file = final_root / "experiment" / "stale.csv"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_bytes(b"stale,1\\n")

    build_final_data(merged_root, final_root)

    assert (final_root / "experiment" / "density.csv").read_bytes() == retained_experiment.read_bytes()
    assert (final_root / "simulation" / "heat_of_vaporization.csv").read_bytes() == retained_simulation.read_bytes()
    assert not stale_file.exists()
    for filename in (
        "thermal_diffusivity.csv",
        "enthalpy.csv",
        "entropy.csv",
        "heat_capacity_at_vapor_saturation_pressure.csv",
        "enthalpy_of_vaporization_or_sublimation.csv",
        "enthalpy_of_transition_or_fusion.csv",
    ):
        assert not (final_root / "experiment" / filename).exists()
