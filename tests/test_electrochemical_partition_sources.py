import hashlib
import io
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

import scripts.crawl_public_properties as crawl
import scripts.structure_public_properties as structure
from scripts.clean_structured_data import clean_non_ilthermo_structured
from scripts.merge_data import merge_data
from scripts.analyze_final_properties import analyze_final_properties


def lethesh_xml(temperatures=(288.15, 298.15), labels=("Ec", "Ea", "ESW"), name="[EMim][TFSI]", values=None):
    values = values or ["−1.8", "2.0", "4.3"] * len(temperatures)
    heads = ''.join(f'<th>{t} K</th>' for t in temperatures)
    targets = ''.join(f'<th>\n{x}\n</th>' for x in labels * len(temperatures))
    cells = ''.join(f'<td>{x}</td>' for x in values)
    return (f'<article><p>The CV scans were performed at 288.15 and 298.15 K in this study.</p>'
            f'<table-wrap><label>TABLE 1</label><table><thead><tr><th>Entry</th><th>ILs</th>{heads}</tr>'
            f'<tr>{targets}</tr></thead><tbody><tr><td>1</td><td>{name}</td>{cells}</tr>'
            '</tbody></table></table-wrap></article>').encode()


def qdb_zip(records):
    buffer = io.BytesIO()
    registry = ET.Element('CompoundRegistry', xmlns='http://www.qsardb.org/QDB')
    values = ['Compound Id\tGas-ionic liquid partition coefficient']
    with zipfile.ZipFile(buffer, 'w') as archive:
        for record in records:
            id_ = record['id']
            node = ET.SubElement(registry, 'Compound')
            for tag, value in {'Id': id_, 'Name': 'solute, IL', 'Description': 'experimental mixture',
                               'InChI': record.get('inchi', Chem.MolToInchi(Chem.MolFromSmiles(record['smiles'])))}.items():
                ET.SubElement(node, tag).text = value
            archive.writestr(f'compounds/{id_}/daylight-smiles', record['smiles'])
            values.append(f"{id_}\t{record['value']}")
        archive.writestr('compounds/compounds.xml', ET.tostring(registry))
        archive.writestr('properties/logK/values', '\n'.join(values))
        archive.writestr('predictions/RF/values', 'M1\t9999')
    return buffer.getvalue()


def archive_fixture(tmp_path, monkeypatch, source, payload):
    files = crawl.EXPERIMENT_ARCHIVES[source]['files']
    if source == 'Lethesh2022':
        pdf = b'%PDF-test'
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as z:
            z.writestr('DataSheet1.pdf', pdf)
        bodies = {'article.xml': payload, 'article.html': b'<html>source</html>', 'supplementary.zip': buf.getvalue()}
        mapping = json.loads((structure.ROOT / 'configs/lethesh2022_ion_map.json').read_text())
        mapping['supplement_pdf_sha256'] = hashlib.sha256(pdf).hexdigest()
        (tmp_path / 'configs').mkdir(exist_ok=True)
        (tmp_path / 'configs/lethesh2022_ion_map.json').write_text(json.dumps(mapping))
        monkeypatch.setattr(structure, 'ROOT', tmp_path)
    else:
        bodies = {'article4.zip': payload, 'repository.html': b'<html>QDB.266</html>', 'toots_thesis.pdf': b'%PDF-definition'}
    monkeypatch.setattr(crawl, 'fetch_bytes', lambda url: bodies[next(n for n, u in files.items() if u == url)])
    return crawl.crawl_source(source, tmp_path / 'raw')


def test_lethesh_measured_temperatures_sign_and_ESW_audit():
    records, rejected = structure.parse_lethesh_table(lethesh_xml())
    assert not rejected
    assert [r['temperature_K'] for r in records] == [288.15, 298.15]
    assert records[0]['cathodic_potential_limit_V'] == -1.8
    assert records[0]['derived_ESW_V'] == 3.8
    assert records[0]['ESW_difference_V'] == -.5
    assert records[0]['ESW_status'] == 'inconsistent'


@pytest.mark.parametrize('changes', [dict(temperatures=(283.15, 298.15)), dict(labels=('Ea', 'Ec', 'ESW'))])
def test_lethesh_refuses_extrapolated_temperatures_or_swapped_headers(changes):
    with pytest.raises(ValueError, match='header mismatch'):
        structure.parse_lethesh_table(lethesh_xml(**changes))


def test_lethesh_nonfinite_rejection_and_rounding():
    records, rejected = structure.parse_lethesh_table(lethesh_xml(values=['NaN', '2', '4', '-2', '2', '4.1']))
    assert len(rejected) == 1
    assert records[0]['ESW_status'] == 'within_rounding'


def test_lethesh_pipeline_conditions_provenance_and_no_ESW_target(tmp_path, monkeypatch):
    source = 'Lethesh2022'
    archive_fixture(tmp_path, monkeypatch, source, lethesh_xml())
    structure.structure_source(source, tmp_path / 'raw', tmp_path / 'structured')
    clean_non_ilthermo_structured(tmp_path / 'structured', tmp_path / 'cleaned')
    merge_data(tmp_path / 'cleaned', tmp_path / 'merged')
    paths = sorted(p.name for p in (tmp_path / 'merged/experiment').glob('*.csv'))
    assert paths == ['anodic_potential_limit.csv', 'cathodic_potential_limit.csv']
    anodic = pd.read_csv(tmp_path / 'merged/experiment/anodic_potential_limit.csv')
    assert anodic['scan_rate_mV/s'].eq(50).all()
    assert anodic.reference_electrode.str.contains('Ag/Ag+ quasi-reference', regex=False).all()
    assert anodic.source_list.eq(source).all()
    assert 'pressure_kPa' not in anodic
    trace = pd.read_csv(tmp_path / 'merged/_audit/public_properties/Lethesh2022/electrochemical_limits_structured_provenance.csv')
    assert trace.status.eq('accepted').all()
    assert trace.raw_IL_name.eq('[EMim][TFSI]').all()
    summary = analyze_final_properties(tmp_path / 'merged', tmp_path / 'analysis', skip_plots=True)
    assert len(summary) == 2
    assert summary.condition_columns.str.contains('reference_electrode').all()


def test_lethesh_unknown_identity_is_not_guessed(tmp_path, monkeypatch):
    payload = lethesh_xml().replace(b'</tbody>',
        b'<tr><td>2</td><td>[unknown][FSI]</td>' + b'<td>-2</td><td>2</td><td>4</td>' * 2 + b'</tr></tbody>')
    archive_fixture(tmp_path, monkeypatch, 'Lethesh2022', payload)
    result = structure.structure_source('Lethesh2022', tmp_path / 'raw', tmp_path / 'structured')
    assert len(result) == 2
    rejected = pd.read_csv(tmp_path / 'structured/Lethesh2022/_audit/structure_rejected.csv')
    assert rejected.identity_issue.eq('No verified mapping').all()


@pytest.mark.parametrize('name,cation,anion_key', [
    ('[Pyr 1,103][TFSI]', 'COCCC[N+]1(C)CCCC1', '[emim][tfsi]'),
    ('[Pyr 1,103][FSI]', 'COCCC[N+]1(C)CCCC1', '[emim][fsi]'),
    ('[N 1,1,1,3][TFSI]', 'CCC[N+](C)(C)C', '[emim][tfsi]'),
    ('[N 2,2,1,102][FSI]', 'COCC[N+](C)(CC)CC', '[emim][fsi]'),
])
def test_lethesh_main_text_resolution_retains_conflicts(tmp_path, monkeypatch, name, cation, anion_key):
    archive_fixture(tmp_path, monkeypatch, 'Lethesh2022', lethesh_xml(name=name))
    result = structure.structure_source('Lethesh2022', tmp_path / 'raw', tmp_path / 'structured')
    mapping = json.loads((structure.ROOT / 'configs/lethesh2022_ion_map.json').read_text())
    assert result.cation.eq(structure.canonicalize_identity_smiles(cation)).all()
    assert result.anion.eq(structure.canonicalize_identity_smiles(mapping['ionic_liquids'][anion_key]['anion'])).all()
    trace = pd.read_csv(tmp_path / 'structured/Lethesh2022/_audit/electrochemical_limits_structured_provenance.csv')
    assert trace.source_conflict.notna().all()
    assert trace.identity_resolution.str.contains('User-approved').all()
    summary = json.loads((tmp_path / 'structured/Lethesh2022/_audit/summary.json').read_text())
    assert summary['source_conflict_conditions'] == len(result)


def test_failure_preserves_last_structured_snapshot(tmp_path, monkeypatch):
    archive_fixture(tmp_path, monkeypatch, 'Lethesh2022', lethesh_xml())
    dest = tmp_path / 'structured/Lethesh2022'
    dest.mkdir(parents=True)
    (dest / 'sentinel').write_text('preserve')
    (tmp_path / 'raw/Lethesh2022/article.xml').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        structure.structure_source('Lethesh2022', tmp_path / 'raw', tmp_path / 'structured')
    assert (dest / 'sentinel').read_text() == 'preserve'


def test_conversion_sign_units_and_standard_state():
    assert structure.logk_to_solvation(1, 298.15) == pytest.approx(-1.3642470130342397)
    ratio = 1000 * 8.31446261815324 * 298.15 / 101325
    correction = structure.logk_to_solvation(0, 298.15, ratio)
    assert correction == pytest.approx(1.89432746)
    assert structure.logk_to_solvation(2, 298.15, ratio) == pytest.approx(-2 * 1.3642470130342397 + correction)
    with pytest.raises(ValueError):
        structure.logk_to_solvation(float('nan'), 298.15)


def test_qdb_roles_values_and_predictions_ignored():
    payload = qdb_zip([{'id': 'M1', 'smiles': '[Cl-].C.CC[N+](C)(C)C', 'value': '1.2'}])
    frame, rejected, raw_count = structure.parse_qdb(payload)
    assert raw_count == len(frame) == 1 and rejected.empty
    assert frame.solute.tolist() == ['C']
    assert frame.logK.tolist() == [1.2]
    assert frame.raw_row_number.tolist() == [2]


def test_duplicate_QDB_property_IDs_are_rejected():
    payload = qdb_zip([{'id': 'M1', 'smiles': '[Cl-].C.CC[N+](C)(C)C', 'value': '1.2'}])
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload)) as original, zipfile.ZipFile(buffer, 'w') as changed:
        for name in original.namelist():
            data = original.read(name)
            if name == 'properties/logK/values':
                data += b'\nM1\t1.3'
            changed.writestr(name, data)
    frame, rejected, raw = structure.parse_qdb(buffer.getvalue())
    assert raw == 2 and frame.empty
    assert rejected.rejection_reason.eq('duplicate_property_ID').all()


@pytest.mark.parametrize('record,reason', [
    ({'id': 'M1', 'smiles': '[Cl-].C.CC[N+](C)(C)C', 'value': 'inf'}, 'Invalid logK'),
    ({'id': 'M1', 'smiles': '[Cl-].C.CC[N+](C)(C)C', 'value': '1', 'inchi': Chem.MolToInchi(Chem.MolFromSmiles('C'))}, 'SMILES_InChI_conflict'),
    ({'id': 'M1', 'smiles': '[Cl-].C.O.CC[N+](C)(C)C', 'value': '1'}, 'ambiguous_component_roles'),
])
def test_qdb_invalid_records_are_traced(record, reason):
    frame, rejected, _ = structure.parse_qdb(qdb_zip([record]))
    assert frame.empty
    assert reason in rejected.rejection_reason.item()
    assert rejected.compound_id.tolist() == ['M1']


@pytest.mark.parametrize('duplicate', [True, False])
def test_qdb_registry_id_errors_are_audited(duplicate):
    payload = qdb_zip([{'id': 'M1', 'smiles': '[Cl-].C.CC[N+](C)(C)C', 'value': '1.2'}])
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload)) as original, zipfile.ZipFile(buffer, 'w') as changed:
        registry = ET.fromstring(original.read('compounds/compounds.xml'))
        if duplicate:
            registry.append(ET.fromstring(ET.tostring(registry[0])))
        else:
            registry[0].remove(registry[0].find('{http://www.qsardb.org/QDB}Id'))
        for member in original.namelist():
            changed.writestr(member, ET.tostring(registry) if member == 'compounds/compounds.xml' else original.read(member))
    frame, rejected, raw = structure.parse_qdb(buffer.getvalue())
    assert raw == 1 and frame.empty
    assert rejected.source_sha256.eq(hashlib.sha256(buffer.getvalue()).hexdigest()).all()
    expected = {'duplicate_registry_ID'} if duplicate else {'missing_registry_ID', 'missing_compound_ID'}
    assert set(rejected.rejection_reason) == expected


def qdb_setup(tmp_path, monkeypatch):
    payload = qdb_zip([{'id': 'M1', 'smiles': '[Cl-].C.CC[N+](C)(C)C', 'value': '1.2'}])
    manifest = archive_fixture(tmp_path, monkeypatch, 'Toots2025_QDB266', payload)
    baseline = tmp_path / 'solvation.csv'
    row = {'cation': 'CC[N+](C)(C)C', 'anion': '[Cl-]', 'solute': 'C', 'temperature_K': 298.15,
           'solvation_kcal/mol': structure.logk_to_solvation(1.2, 298.15), 'source_list': 'existing; second'}
    pd.DataFrame([row, {**row, 'temperature_K': 300.0}]).to_csv(baseline, index=False)
    return payload, baseline, manifest


def test_qdb_closed_gate_overlap_temperature_isolation_and_audit_only(tmp_path, monkeypatch):
    _, baseline, _ = qdb_setup(tmp_path, monkeypatch)
    result = structure.structure_source('Toots2025_QDB266', tmp_path / 'raw', tmp_path / 'structured', baseline)
    assert result.empty
    source_root = tmp_path / 'structured/Toots2025_QDB266'
    assert not list(source_root.glob('*.csv'))
    summary = json.loads((source_root / '_audit/summary.json').read_text())
    assert summary['overlap_pairs'] == 1
    assert not summary['merge_allowed'] and summary['actual_new_systems'] == 0
    groups = pd.read_csv(source_root / '_audit/residual_groups.csv')
    groups = groups.loc[groups.group_type.eq('existing_source')]
    assert set(groups['group'].map(lambda value: json.loads(value)[0])) == {'existing', 'second'}
    assert groups.n.eq(1).all()
    clean_non_ilthermo_structured(tmp_path / 'structured', tmp_path / 'cleaned')
    merge_data(tmp_path / 'cleaned', tmp_path / 'merged')
    assert (tmp_path / 'merged/_audit/public_properties/Toots2025_QDB266/summary.json').is_file()
    assert not (tmp_path / 'merged/experiment/solvation.csv').exists()


def test_reviewed_gate_is_input_bound_and_reuses_merge_provenance(tmp_path, monkeypatch):
    payload, baseline, _ = qdb_setup(tmp_path, monkeypatch)
    evidence = {'approved': True, 'archive_sha256': hashlib.sha256(payload).hexdigest(),
                'baseline_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(), 'log_base': 10,
                'partition_definition': 'c_IL/c_gas', 'temperature_K': 298.15, 'infinite_dilution': True, 'unresolved_issues': [],
                'reviewer': 'fixture reviewer', 'existing_standard_state_evidence': 'fixture 1M/1M documentation',
                'qdb_definition_evidence': 'fixture concentration-ratio definition', 'anomaly_resolution': 'fixture has zero residuals',
                'liquid_to_gas_standard_concentration_ratio': 1}
    evidence_file = tmp_path / 'evidence.json'
    evidence_file.write_text(json.dumps(evidence))
    result = structure.structure_source('Toots2025_QDB266', tmp_path / 'raw', tmp_path / 'structured', baseline, evidence_file)
    assert len(result) == 1
    clean_non_ilthermo_structured(tmp_path / 'structured', tmp_path / 'cleaned')
    dest = tmp_path / 'cleaned/AIonopedia'
    dest.mkdir()
    pd.read_csv(baseline).drop(columns='source_list').to_csv(dest / 'solvation.csv', index=False)
    merge_data(tmp_path / 'cleaned', tmp_path / 'merged')
    merged = pd.read_csv(tmp_path / 'merged/experiment/solvation.csv')
    assert len(merged) == 2
    assert merged.loc[merged.temperature_K.eq(298.15), 'source_list'].item() == 'AIonopedia; Toots2025_QDB266'
    evidence['archive_sha256'] = 'stale'
    frame, _, _ = structure.parse_qdb(payload)
    assert not structure.qdb_merge_evidence(evidence, hashlib.sha256(payload).hexdigest(), evidence['baseline_sha256'], frame)[0]


def test_stale_cleaned_QDB_labels_cannot_bypass_gate(tmp_path):
    source = tmp_path / 'cleaned/Toots2025_QDB266'
    source.mkdir(parents=True)
    pd.DataFrame([{'solute': 'C', 'solvation_kcal/mol': -1}]).to_csv(source / 'solvation.csv', index=False)
    previous = tmp_path / 'merged/experiment'
    previous.mkdir(parents=True)
    (previous / 'solvation.csv').write_text('original final input')
    with pytest.raises(ValueError, match='publication blocked'):
        merge_data(tmp_path / 'cleaned', tmp_path / 'merged')
    assert (previous / 'solvation.csv').read_text() == 'original final input'


def test_reference_conditions_are_distinct_merge_keys(tmp_path):
    source = tmp_path / 'cleaned/Lethesh2022'
    source.mkdir(parents=True)
    row = {'cation': 'C[N+](C)(C)C', 'anion': '[Cl-]', 'temperature_K': 298.15,
           'reference_electrode': 'Ag/Ag+', 'working_electrode': 'Pt', 'scan_rate_mV/s': 50,
           'anodic_potential_limit_V': 2}
    pd.DataFrame([row, {**row, 'reference_electrode': 'Fc/Fc+'}, {**row, 'scan_rate_mV/s': 100}]).to_csv(source / 'potentials.csv', index=False)
    merge_data(tmp_path / 'cleaned', tmp_path / 'merged')
    assert len(pd.read_csv(tmp_path / 'merged/experiment/anodic_potential_limit.csv')) == 3


def test_potential_tasks_share_grouped_folds():
    import scripts.build_training_splits as splits
    groups = pd.Series([f'IL-{i}' for i in range(10) for _ in range(2)])
    folds = []
    for name in ('anodic', 'cathodic'):
        namespace = splits.stage3_group_split_namespace(f'experiment/{name}_potential_limit')
        folds.append(splits.task_group_kfold_assignments(
            groups, task_id=namespace, strategy='IL', repeats=5, seed=42))
    for ea, ec in zip(*folds):
        pd.testing.assert_series_equal(ea, ec)
        assert ea.groupby(groups).nunique().eq(1).all()
        assert set(ea) == set(range(5))
    assert splits.stage3_group_split_namespace('experiment/solvation') == 'experiment/solvation'
