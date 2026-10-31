import hashlib
import json

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from scripts import crawl_public_properties as crawl
from scripts.build_final_data import build_final_data
from scripts.clean_structured_data import clean_non_ilthermo_structured
from scripts.merge_data import merge_data, water_gamma_to_solvation, water_saturation_pressure_pa
from scripts.structure_public_properties import structure_source


CATION = 'CC[n+]1ccn(C)c1'
ANION = 'F[B-](F)(F)F'


def water_snapshot(tmp_path, monkeypatch, records=None):
    rows = records or [dict(Cation_SMILES=CATION, Anion_SMILES=ANION, Mole_frac_water=x,
                            Temperature=298.15, Pressure=101.0, Activityt_water=gamma,
                            Ref='original measurement', **{'Sl-No': i, 'Delta[Act]_comp0': .01})
                       for i, (x, gamma) in enumerate([(0., 2.), (.2, 1.5), (1., 1.)])]
    def payload(url):
        return b'%PDF-1.4\nreference' if url.endswith('.pdf') else pd.DataFrame(rows).to_csv(index=False).encode()
    monkeypatch.setattr(crawl, 'fetch_bytes', payload)
    crawl.crawl_source('WaterActivity', tmp_path / 'raw')
    structure_source('WaterActivity', tmp_path / 'raw', tmp_path / 'structured')
    clean_non_ilthermo_structured(tmp_path / 'structured', tmp_path / 'cleaned', ['WaterActivity'])
    return rows


def density(tmp_path, values=(1.2,), temperature=298.15, pressure=101.0, cation=CATION):
    root = tmp_path / 'cleaned/AIonopedia'
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(cation=cation, anion=ANION, temperature_K=temperature, pressure_kPa=pressure,
                       **{'density_g/cm^3': v}) for v in values]).to_csv(root / 'density.csv', index=False)


def audit(tmp_path):
    root = tmp_path / 'merged/_audit/water_solvation'
    return pd.read_csv(root / 'conversion_records.csv'), json.loads((root / 'summary.json').read_text())


def test_physical_sign_units_and_reference_vapor_pressure():
    # Independent reference: IAPWS saturation table at 25 C and 100 C.
    assert water_saturation_pressure_pa(298.15) == pytest.approx(3169.8, rel=1e-4)
    assert water_saturation_pressure_pa(373.15) == pytest.approx(101418, rel=1e-4)
    # Water self-solvation using its 25 C density, equal-concentration standards.
    self_solvation = water_gamma_to_solvation(1., 298.15, 18.01528, .997047)
    assert self_solvation == pytest.approx(-6.32, abs=.01)
    assert water_gamma_to_solvation(2., 298.15, 18.01528, .997047) - self_solvation == pytest.approx(.41068, abs=1e-4)
    assert water_gamma_to_solvation(1., 298.15, 18.01528, .4985235) - self_solvation == pytest.approx(.41068, abs=1e-4)


@pytest.mark.parametrize('temperature', [273., 647.096, float('nan')])
def test_no_vapor_pressure_extrapolation(temperature):
    with pytest.raises(ValueError):
        water_saturation_pressure_pa(temperature)


def test_pipeline_only_converts_infinite_dilution_and_preserves_dependencies(tmp_path, monkeypatch):
    water_snapshot(tmp_path, monkeypatch)
    density(tmp_path, values=(1.0, 1.2, 1.2, 1.4))
    merge_data(tmp_path / 'cleaned', tmp_path / 'merged')
    output = pd.read_csv(tmp_path / 'merged/experiment/solvation.csv')
    assert list(output) == ['cation', 'anion', 'solute', 'temperature_K', 'solvation_kcal/mol', 'source_list']
    assert len(output) == 1 and output.solute.item() == 'O' and output.source_list.item() == 'WaterActivity'
    records, summary = audit(tmp_path)
    assert records.status.tolist().count('published') == 1
    assert records.status.tolist().count('finite_composition_not_converted') == 2
    row = records.loc[records.status.eq('published')].iloc[0]
    assert row.density_g_cm3 == 1.2 and row.density_unique_values == 3
    assert row.raw_row_number == 2 and row.raw_gamma_uncertainty == .01
    assert row.reference == 'original measurement'
    matches = pd.read_csv(tmp_path / 'merged/_audit/water_solvation/density_matches.csv')
    assert len(matches) == 4 and matches.density_source.eq('AIonopedia').all()
    for relative, digest in summary['input_hashes'].items():
        assert hashlib.sha256((tmp_path / 'cleaned' / relative).read_bytes()).hexdigest() == digest
    # Raw gamma remains in merged/audit, but is retired from final publication.
    charge = tmp_path / 'charge_20260514'
    charge.mkdir()
    pd.DataFrame([dict(cation=CATION, anion=ANION, solute='O', temperature_K=298.15,
                       source_list='transfer source', **{'transfer_kcal/mol': -1.})]).to_csv(
        tmp_path / 'merged/experiment/transfer.csv', index=False)
    pd.DataFrame(columns=['mol_id', 'SMILES', 'charge', 'source_list']).to_csv(tmp_path / 'merged/simulation/charge.csv', index=False)
    build_final_data(tmp_path / 'merged', tmp_path / 'final', charge)
    assert not (tmp_path / 'final/experiment/water_activity_coefficient.csv').exists()
    assert (tmp_path / 'final/experiment/solvation.csv').exists()
    assert (tmp_path / 'final/_audit/water_solvation/summary.json').exists()
    hydration = pd.read_csv(tmp_path / 'final/experiment/hydration.csv')
    assert hydration['hydration_kcal/mol'].item() == pytest.approx(output['solvation_kcal/mol'].item() - 1.)


@pytest.mark.parametrize('temperature,pressure,expected', [
    (303.15, 101., 'missing_same_temperature_density'),
    (298.15, 500., 'missing_near_atmospheric_density'),
    (298.15, 101.325, 'published'),
])
def test_density_temperature_and_pressure_policy(tmp_path, monkeypatch, temperature, pressure, expected):
    water_snapshot(tmp_path, monkeypatch)
    density(tmp_path, temperature=temperature, pressure=pressure)
    merge_data(tmp_path / 'cleaned', tmp_path / 'merged')
    records, _ = audit(tmp_path)
    assert records.loc[records.x_water_unitless.eq(0), 'status'].item() == expected
    if expected == 'published':
        row = records.loc[records.status.eq('published')].iloc[0]
        assert row.density_match == 'exact_T_near_atmospheric_P'
        assert row.density_pressure_difference_kPa == pytest.approx(.325)


def test_exact_pressure_wins_over_more_density_observations(tmp_path, monkeypatch):
    water_snapshot(tmp_path, monkeypatch)
    density(tmp_path, values=(1.2,))
    root = tmp_path / 'cleaned/ILBERT'
    root.mkdir()
    pd.DataFrame([dict(cation=CATION, anion=ANION, temperature_K=298.15, pressure_kPa=101.325,
                       **{'density_g/cm^3': v}) for v in [1., 2., 3.]]).to_csv(root / 'density.csv', index=False)
    merge_data(tmp_path / 'cleaned', tmp_path / 'merged')
    records, _ = audit(tmp_path)
    row = records.loc[records.status.eq('published')].iloc[0]
    assert row.density_g_cm3 == 1.2 and row.density_match == 'exact_T_P'


@pytest.mark.parametrize('offset,expected', [(0., 'published'), (.5, 'overlap_conflict_IL_quarantined')])
def test_overlap_no_offsets_or_overwriting_old_labels(tmp_path, monkeypatch, offset, expected):
    water_snapshot(tmp_path, monkeypatch)
    density(tmp_path)
    mass = sum(Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in [CATION, ANION])
    value = water_gamma_to_solvation(2., 298.15, mass, 1.2) + offset
    pd.DataFrame([dict(cation=CATION, anion=ANION, solute='O', temperature_K=298.15,
                       **{'solvation_kcal/mol': value})]).to_csv(tmp_path / 'cleaned/AIonopedia/solvation.csv', index=False)
    merge_data(tmp_path / 'cleaned', tmp_path / 'merged')
    records, _ = audit(tmp_path)
    assert records.loc[records.x_water_unitless.eq(0), 'status'].item() == expected
    output = pd.read_csv(tmp_path / 'merged/experiment/solvation.csv')
    assert len(output) == 1 and output['solvation_kcal/mol'].item() == pytest.approx(value)
    assert ('WaterActivity' in output.source_list.item()) == (offset == 0)
    overlap = pd.read_csv(tmp_path / 'merged/_audit/water_solvation/overlap_pairs.csv')
    assert overlap.residual_kcal_mol.item() == pytest.approx(-offset)


def test_stale_water_snapshot_preserves_existing_merged(tmp_path, monkeypatch):
    water_snapshot(tmp_path, monkeypatch)
    path = tmp_path / 'cleaned/WaterActivity/water_activity_coefficient_structured.csv'
    path.write_text(path.read_text().replace('2.0', '3.0'))
    output = tmp_path / 'merged/experiment'
    output.mkdir(parents=True)
    (output / 'solvation.csv').write_text('old')
    with pytest.raises(ValueError, match='conversion blocked'):
        merge_data(tmp_path / 'cleaned', tmp_path / 'merged')
    assert (output / 'solvation.csv').read_text() == 'old'


def test_retirement_without_conversion_audit_preserves_final(tmp_path):
    merged = tmp_path / 'merged/experiment'
    merged.mkdir(parents=True)
    (merged / 'water_activity_coefficient.csv').write_text('unconverted')
    final = tmp_path / 'final'
    final.mkdir()
    (final / 'old').write_text('keep')
    with pytest.raises(ValueError, match='requires a current conversion audit'):
        build_final_data(tmp_path / 'merged', final)
    assert (final / 'old').read_text() == 'keep'


def test_conflicting_IL_is_quarantined_at_other_temperatures_too(tmp_path, monkeypatch):
    rows = [dict(Cation_SMILES=CATION, Anion_SMILES=ANION, Mole_frac_water=0.,
                 Temperature=t, Pressure=101., Activityt_water=2., Ref='measurement')
            for t in [298.15, 303.15]]
    water_snapshot(tmp_path, monkeypatch, rows)
    density(tmp_path)
    path = tmp_path / 'cleaned/AIonopedia/density.csv'
    pd.concat([pd.read_csv(path), pd.read_csv(path).assign(temperature_K=303.15)]).to_csv(path, index=False)
    pd.DataFrame([dict(cation=CATION, anion=ANION, solute='O', temperature_K=298.15,
                       **{'solvation_kcal/mol': 0.})]).to_csv(path.parent / 'solvation.csv', index=False)
    merge_data(tmp_path / 'cleaned', tmp_path / 'merged')
    records, summary = audit(tmp_path)
    assert records.status.eq('overlap_conflict_IL_quarantined').all()
    assert summary['published_records_before_merge'] == 0 and summary['conflicting_ILs'] == 1
    assert summary['before'] == summary['after']


def test_multivalent_formula_not_silently_treated_as_one_to_one(tmp_path, monkeypatch):
    cation = 'C[N+](C)(C)CC[N+](C)(C)C'
    rows = [dict(Cation_SMILES=cation, Anion_SMILES=ANION, Mole_frac_water=0.,
                 Temperature=298.15, Pressure=101., Activityt_water=2., Ref='measurement')]
    water_snapshot(tmp_path, monkeypatch, rows)
    density(tmp_path, cation=cation)
    merge_data(tmp_path / 'cleaned', tmp_path / 'merged')
    records, _ = audit(tmp_path)
    assert records.status.item() == 'unsupported_ion_stoichiometry'
    assert not (tmp_path / 'merged/experiment/solvation.csv').exists()


def test_empty_cleaned_snapshot_has_auditable_zero_conversion(tmp_path, monkeypatch):
    rows = [dict(Cation_SMILES=CATION, Anion_SMILES=ANION, Mole_frac_water=0.,
                 Temperature=298.15, Pressure=101., Activityt_water=-1., Ref='measurement')]
    water_snapshot(tmp_path, monkeypatch, rows)
    merge_data(tmp_path / 'cleaned', tmp_path / 'merged')
    records, summary = audit(tmp_path)
    assert records.empty and summary['input_records'] == summary['published_records_before_merge'] == 0


def test_reference_hash_failure_preserves_previous_outputs(tmp_path, monkeypatch):
    water_snapshot(tmp_path, monkeypatch)
    (tmp_path / 'cleaned/WaterActivity/_audit/water_saturation_iapws1992.pdf').write_text('changed')
    previous = tmp_path / 'merged/experiment'
    previous.mkdir(parents=True)
    (previous / 'solvation.csv').write_text('keep')
    with pytest.raises(ValueError, match='reference hash mismatch'):
        merge_data(tmp_path / 'cleaned', tmp_path / 'merged')
    assert (previous / 'solvation.csv').read_text() == 'keep'
