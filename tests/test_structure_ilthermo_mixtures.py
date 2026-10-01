import json

import pandas as pd
import pytest

from scripts import structure_ilthermo_mixtures as s
from scripts import crawl_ilthermo_mixtures as c


def fixture_entry(size):
    return {'title': 'Density', 'phases': ['Liquid'], 'ref': {'full': 'ref', 'title': 'paper'},
            'expmeth': 'test', 'solvent': None, 'constr': [], 'footer': '',
            'components': [{'idout': f'c{i}', 'name': name, 'formula': '', 'mw': '', 'sample': []}
                           for i, name in enumerate(['1,3-compound', 'water', 'ethanol'][:size], 1)],
            'dhead': [['Temperature, K', None], ['Mole fraction of 1,3-compound', 'Liquid'],
                      ['Density, kg/m<SUP>3</SUP>', 'Liquid']],
            'data': [[['300'], ['0.2'], ['1000', '0.1']], [['301'], None]]}


def make_snapshot(root):
    for size, eid in [(2, 'b1'), (3, 't1')]:
        c.atomic_json(c.manifest_path(root, size), {'n_compounds': size, 'rows': [
            {'id': eid, 'num_components': size, 'num_data_points': 2, 'property': 'Density', 'reference': 'ref', 'phases': 'Liquid'}]})
        c.save_entry(root, eid, fixture_entry(size))


def test_headers_and_mapping(monkeypatch):
    comps = [{'component_index': 1, 'name': '1,3-compound'}, {'component_index': 2, 'name': 'water'}]
    h = s.header_metadata('Mole fraction of 1,3-compound', comps)
    assert s.header_metadata('Mole fraction of water<SUP>*</SUP>', comps)[3] == 2
    assert h[0:4] == ('Mole fraction of 1,3-compound', 'Mole fraction of 1,3-compound', None, 1)
    assert s.header_metadata('MolaLity of 1,3-compound, mol/kg', comps)[2:5] == ('mol/kg', 1, 'molality')
    assert s.header_metadata('Weight fraction of missing', comps)[-1] == 'unmatched_component'
    assert s.header_metadata('Mole fraction of water', comps + [comps[1]])[-1] == 'ambiguous_component'
    monkeypatch.setattr(s._compounds, 'id2smiles', {'id': 'O', 'bad': 'invalid'})
    monkeypatch.setattr(s._compounds, 'name2smiles', {'water': 'CCO'})
    s.molecule.cache_clear()
    assert s.molecule('id', 'water')['smiles'] == 'O'
    assert s.molecule('unknown', 'water')['smiles_match_source'] == 'name'
    assert s.molecule('bad', 'water')['smiles_status'] == 'invalid'
    assert s.molecule('unknown', 'missing')['smiles_status'] == 'unmatched'
    s.molecule.cache_clear()


def obs(name, value, unit=None, phase=None, index=1, **kwargs):
    return dict(variable_name=name, value_numeric=s.number(value), value_raw=value, unit=unit, unit_raw=unit,
                phase=phase, variable_index=index, data_point_index=1, raw_header=name, uncertainty_raw='0.1',
                composition_kind=None, component_index=None, composition_scope=None, metadata_status='ok', **kwargs)


def test_phase_context_and_conversions():
    entry = dict(entry_id='test', mixture_size=2, validation_status='ok', phases_json='["L1", "L2"]',
                 reference_full='', reference_title='', raw_sha256='hash')
    target = obs('Density', '1000', 'kg/m^3', 'L1', 3)
    point = [obs('Temperature', '300', 'K'), obs('Pressure', '100', 'kPa', 'L2'),
             obs('Wavelength', '5000', 'A', 'L1'), target]
    row = s.training_row(entry, [], point, target, s.ILTHERMO_SPECS['density'], lambda *args: None)
    assert row['density_g/cm^3'] == 1
    assert row['temperature_K'] == 300
    assert row['wavelength_nm'] == 500
    assert row.get('pressure_kPa') is None
    assert row['target_uncertainty_raw'] == '0.1'
    assert s.target_conversion(s.ILTHERMO_SPECS['viscosity'], obs('Viscosity', '0.01', 'Pa*s')) == 1
    with pytest.raises(ValueError):
        s.target_conversion(s.ILTHERMO_SPECS['viscosity'], obs('Kinematic viscosity', '1', 'm^2/s'))
    row = s.training_row(entry, [], [obs('Temperature', '300', 'unknown'), target], target,
                         s.ILTHERMO_SPECS['density'], lambda *args: None)
    assert row['temperature_K'] is None
    assert 'conversion_failed:temperature_K' in row['quality_flags']


def test_pipeline_sparse_atomic_and_repeat(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    make_snapshot(source)
    report = s.run(source, output)
    assert (report['entries'], report['components'], report['data_points'], report['observations']) == (2, 5, 4, 10)
    observations = pd.read_parquet(output / 'all/observations.parquet')
    assert observations.loc[observations.variable_index == 2, 'unit'].isna().all()
    assert observations.value_raw.isna().sum() == 2
    components = pd.read_parquet(output / 'all/components.parquet')
    assert components.loc[components.entry_id == 't1', 'component_index'].tolist() == [1, 2, 3]
    frame = pd.read_csv(output / 'properties/density.csv')
    assert frame.sample_id.is_unique and len(frame) == 2
    assert (frame['density_g/cm^3'] == 1).all()
    assert frame.quality_flags.str.contains('source_irregular').all()
    assert s.run(source, output)['input_sha256'] == report['input_sha256']
    before = (output / 'all/observations.parquet').read_bytes()
    (source / 'bronze/entries/b1.json').write_text('{}')
    with pytest.raises(ValueError, match='Bronze'):
        s.run(source, output)
    assert (output / 'all/observations.parquet').read_bytes() == before
    assert not list(output.glob('.structure-*'))


def test_reported_composition_phase_and_unknown_target():
    entry = dict(entry_id='phase', mixture_size=3, validation_status='ok', phases_json='["L1", "L2"]',
                 reference_full='', reference_title='', raw_sha256='hash')
    target = obs('Density', '1000', 'unknown', 'L1', 9)
    x1 = obs('Mole fraction of first', '0.2', phase='L1', index=2)
    x1.update(composition_kind='mole_fraction', component_index=1)
    x2 = obs('Mole fraction of first', '0.8', phase='L2', index=3)
    x2.update(composition_kind='mole_fraction', component_index=1)
    mass = obs('Weight fraction of second', '0.4', phase='L1', index=4)
    mass.update(composition_kind='mass_fraction', component_index=2)
    molality = obs('MolaLity of third', '2', 'mol/kg', 'L1', 5)
    molality.update(composition_kind='molality', component_index=3)
    row = s.training_row(entry, [], [x1, x2, mass, molality, target], target,
                         s.ILTHERMO_SPECS['density'], lambda *args: None)
    assert row['component_1_mole_fraction'] == 0.2
    assert row.get('component_2_mole_fraction') is None
    assert row['component_2_mass_fraction'] == 0.4
    assert row['component_3_molality_mol/kg'] == 2
    assert row['density_g/cm^3'] is None
    assert 'conversion_failed:density_g/cm^3' in row['quality_flags']
    assert 'incomplete_reported_composition:mole_fraction' in row['quality_flags']
