import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import image as mpl_image

from scripts import analyze_dataset_relationship_graph as graph


EXPECTED_METRICS = {
    'spearman', 'distance_correlation', 'binary_i_over_h', 'multiclass_mi',
    'predictability_cv_nmae',
}


@pytest.fixture
def config():
    return graph.load_config(graph.DEFAULT_CONFIG)


def frame(y, t=None, p=None):
    d = pd.DataFrame({'_value': y})
    if t is not None:
        d['temperature_K'] = t
    if p is not None:
        d['pressure_kPa'] = p
    return d


def test_raw_pending_and_zero_df(config):
    refs = config['references']
    single = graph.fit_signature(frame([7], [350]), 'pending_formula', refs)
    assert single['signature'] == 7 and single['signature_kind'] == 'single_raw'
    assert graph.fit_signature(frame([7, 8], [300, 320]), 'pending_formula', refs)['status'] == 'pending_formula'
    fit = graph.fit_signature(frame([2, 8], [298.15, 328.15]), 'linear_temperature', refs)
    assert fit['signature'] == pytest.approx(2)
    assert fit['residual_df'] == 0 and np.isnan(fit['uncertainty'])


def test_formula_recovery_and_rank(config):
    refs = config['references']
    t = np.array([300, 310, 320, 330.])
    p = np.array([100, 300, 120, 400.])
    y = np.exp(.2 - .002 * (t - refs['temperature_K']) + 1e-5 * (p - refs['pressure_kPa']))
    fit = graph.fit_signature(frame(y, t, p), 'log_density', refs)
    assert fit['signature'] == pytest.approx(np.exp(.2))
    assert fit['extrapolated']
    assert fit['status'] == 'ok'
    collinear = graph.fit_signature(frame(y, t, 2*t), 'linear_temperature_pressure', refs)
    assert collinear['status'] == 'insufficient_condition_design'
    constant = graph.fit_signature(frame([2, 4], [298.15, 308.15], [200, 200]), 'linear_temperature_pressure', refs)
    assert constant['signature'] == pytest.approx(2)
    assert constant['reference_mismatch']
    assert json.loads(constant['actual_reference'])['pressure_kPa'] == 200


def test_inverse_and_cauchy(config):
    refs = config['references'];t = np.array([290., 300, 320, 350, 380, 400])
    y = 5 + 1000 * (1/t - 1/refs['temperature_K'])
    assert graph.fit_signature(frame(y, t), 'inverse_temperature', refs)['signature'] == pytest.approx(5)
    lam = np.array([450., 500, 550, 600, 650, 700]);p = np.array([100., 150, 100, 300, 100, 400])
    y = 1.4 + .001*(t-refs['temperature_K']) + .00001*(p-refs['pressure_kPa']) + 3000*(lam**-2 - 589.**-2)
    d = frame(y,t,p);d['wavelength_nm']=lam
    assert graph.fit_signature(d, 'cauchy', refs)['signature'] == pytest.approx(1.4)


def test_reference_frequency_and_x_co2_formulas(config):
    refs = config['references']
    dynamic = pd.DataFrame({
        '_value': [8., 6., 99.], 'frequency_MHz': [10000., 10000., 9000.],
        'temperature_K': [298.15, 298.15, 298.15],
        'pressure_kPa': [101.325, 101.325, 101.325],
    })
    fit = graph.fit_signature(dynamic, 'reference_frequency_10ghz', refs)
    assert fit['signature'] == 7
    assert fit['fit_method'] == 'reference_frequency_median'
    missing = graph.fit_signature(dynamic.iloc[[2]], 'reference_frequency_10ghz', refs)
    assert missing['status'] == 'missing_reference_frequency'
    assert np.isnan(missing['signature'])

    temperature = np.array([285., 300., 320., 340., 300., 330.])
    pressure = np.array([100., 250., 140., 400., 500., 220.])
    a, b, c = 10., 800., .015
    transformed = a + b * (1 / temperature - 1 / refs['temperature_K']) \
        + c * (pressure - refs['pressure_kPa']) / temperature
    x_co2 = pressure * np.exp(-transformed)
    data = frame(x_co2, temperature, pressure)
    fit = graph.fit_signature(data, 'x_co2_reference_solubility', refs)
    assert fit['signature'] == pytest.approx(refs['pressure_kPa'] * np.exp(-a))
    assert json.loads(fit['parameters']) == pytest.approx({
        'a': a, 'b_inverse_temperature': b, 'c_pressure_over_temperature': c,
    })
    assert json.loads(fit['fit_quality'])['retained_terms'] == [
        'b_inverse_temperature', 'c_pressure_over_temperature',
    ]
    invalid = frame([1.], [298.15], [101.325])
    assert graph.fit_signature(invalid, 'x_co2_reference_solubility', refs)['status'] == 'invalid_x_co2'

    collinear_temperature = np.array([290., 300., 310., 320.])
    constant_pressure = np.full(4, 200.)
    response = 9 + 500 * (1 / collinear_temperature - 1 / refs['temperature_K'])
    collinear_x = constant_pressure * np.exp(-response)
    fit = graph.fit_signature(frame(collinear_x, collinear_temperature, constant_pressure),
                              'x_co2_reference_solubility', refs)
    quality = json.loads(fit['fit_quality'])
    assert quality['retained_terms'] == ['b_inverse_temperature']
    assert quality['dropped_terms'] == ['c_pressure_over_temperature']


def test_vft_recovery_and_insufficient(config):
    refs=config['references'];t=np.linspace(280,400,15)
    y=2+400*(1/(t-150)-1/(refs['temperature_K']-150))
    fit=graph.fit_signature(frame(y,t,np.full(len(t),101.325)), 'vft', refs)
    assert fit['status']=='ok'
    assert fit['signature']==pytest.approx(2,abs=1e-5)
    assert json.loads(fit['parameters'])['T0']==pytest.approx(150,abs=1e-3)
    assert graph.fit_signature(frame(y[:3],t[:3],[100]*3),'vft',refs)['status']=='insufficient_vft_design'


def test_arrhenius_temperature_pressure_recovery(config):
    refs = config['references']
    temperature = np.array([280., 300., 320., 340., 360.])
    pressure = np.array([100., 250., 120., 400., 180.])
    y = 2.5 + 700 * (1 / temperature - 1 / refs['temperature_K']) \
        + .002 * (pressure - refs['pressure_kPa'])
    fit = graph.fit_signature(frame(y, temperature, pressure),
                              'arrhenius_temperature_pressure', refs)
    assert fit['signature'] == pytest.approx(2.5)
    assert fit['status'] == 'ok'
    assert json.loads(fit['parameters']) == pytest.approx({
        'intercept': 2.5, 'temperature_K': 700., 'pressure_kPa': .002,
    })
    two_temperature = np.array([290., 310.])
    two_y = 2.5 + 700 * (1 / two_temperature - 1 / refs['temperature_K'])
    two_fit = graph.fit_signature(frame(two_y, two_temperature, [101.325, 101.325]),
                                  'arrhenius_temperature_pressure', refs)
    assert two_fit['signature'] == pytest.approx(2.5)
    assert two_fit['residual_df'] == 0 and np.isnan(two_fit['uncertainty'])
    ambiguous = graph.fit_signature(frame(two_y, two_temperature, [100., 200.]),
                                    'arrhenius_temperature_pressure', refs)
    assert ambiguous['status'] == 'insufficient_condition_design'
    assert np.isnan(ambiguous['signature'])
    selected = graph.fit_signature(
        frame([-0.661544, -0.520857], [298.15, 313.], [100., 101.325]),
        'arrhenius_temperature_pressure', refs)
    assert selected['signature'] == pytest.approx(-0.661544)
    assert selected['signature_kind'] == 'reference_temperature_observation'
    assert selected['reference_mismatch']
    for source in ('experiment/viscosity.csv', 'experiment/electrical_conductivity.csv',
                   'experiment/self_diffusion_coefficient.csv'):
        assert config['formulas'][source] == 'arrhenius_temperature_pressure'


def test_solute_identification_and_components(config):
    d=pd.DataFrame([{'cation':c,'anion':'a','solute':s,'temperature_K':298.15,'_value':effect+se}
                    for c,effect in [('c1',2),('c2',5)] for s,se in [('s1',-3),('s2',3)]])
    node={'formula':'solute_only'}
    rows,units=graph.solute_signatures(d,node,config['references'])
    assert [r['signature'] for r in rows]==pytest.approx([2,5])
    assert len(units)==4
    extra = d.iloc[[0]].copy()
    extra['temperature_K'] = 308.15
    extra['_value'] += 1
    mixed, _ = graph.solute_signatures(pd.concat([d, extra]), {'formula': 'solute_linear_temperature'}, config['references'])
    assert mixed[0]['mixed_unit_signature_kinds']
    assert set(json.loads(mixed[0]['unit_signature_kinds'])) == {'single_raw', 'condition_corrected'}
    isolated=pd.DataFrame([{'cation':'c3','anion':'a','solute':'s3','temperature_K':298.15,'_value':9}])
    rows,_=graph.solute_signatures(pd.concat([d,isolated]),node,config['references'])
    assert rows[-1]['status']=='unidentifiable_solute_component'
    assert np.isnan(rows[-1]['signature'])


def test_metrics_symmetry_pair_bins_small_and_ties():
    rng=np.random.default_rng(2);x=rng.normal(size=100);y=x*x+.1*rng.normal(size=100)
    a,_=graph.metric_values(x,y);b,_=graph.metric_values(y,x)
    for m in graph.SYMMETRIC:
        assert a[m]==pytest.approx(b[m])
    small,states=graph.metric_values(np.array([1.,2]),np.array([3.,4]))
    assert small['binary_i_over_h']==pytest.approx(1)
    assert 'binary_mi' not in small
    assert states['spearman']=='insufficient_samples'
    vals,states=graph.metric_values(np.ones(30),np.ones(30))
    assert np.isnan(vals['binary_i_over_h']) and states['binary_i_over_h']=='zero_target_entropy'
    assert states['multiclass_mi']=='degenerate_quantile_bins'
    _,states=graph.metric_values(np.array([]),np.array([]))
    assert set(states.values())=={'no_shared_signatures'}
    first,_=graph.metric_values(np.arange(30.),np.arange(30.))
    assert first['binary_i_over_h']==pytest.approx(1)
    assert graph.quantile_labels(np.ones(30),3) is None


def test_cv_and_train_fold_scaling(monkeypatch):
    x=np.linspace(-2,2,50);y=x*x
    a,reason=graph.cv_nmae(x,y,42)
    assert reason=='ok' and 0<a<1
    assert graph.cv_nmae(x,y,42)[0]==a
    assert graph.cv_nmae(x,np.ones(50),42)[1]=='zero_baseline_error'
    original=graph.StandardScaler.fit
    seen=[]
    def fit(self,X,*args,**kwargs):
        seen.append(len(X));return original(self,X,*args,**kwargs)
    monkeypatch.setattr(graph.StandardScaler,'fit',fit)
    graph.cv_nmae(x,y,42)
    assert seen==[40]*5


def fixture_root(tmp_path):
    root=tmp_path/'splits';root.mkdir()
    entries=[]
    for stage,task,source,targets,system_type,repeats in [
        (2,'simulation/a','simulation/a.csv','a;b','il',1),
        (3,'experiment/b','experiment/b.csv','b','il',2),
        (3,'experiment/c','experiment/c.csv','b','il',1),
        (2,'simulation/qm','simulation/qm.csv','q1;q2','molecule',1)]:
        materialized=f'stage{stage}/{task.split("/")[-1]}'
        entries.append(dict(stage=stage,task_id=task,source_file=source,target_columns=targets,
                            system_type=system_type,materialized_path=materialized,repeats=repeats,condition_columns=''))
        if system_type!='il':continue
        directory=root/materialized;directory.mkdir(parents=True)
        fold_dir = 'IL/cv1' if repeats > 1 else 'IL'
        names=['train.csv','valid.csv'] if stage==2 else [f'{fold_dir}/fold{i}.csv' for i in range(1,6)]
        for i,name in enumerate(names):
            p=directory/name;p.parent.mkdir(parents=True,exist_ok=True)
            pd.DataFrame({'cation':[f'c{i}'],'anion':['a'],'a':[i+1],'b':[i+2]}).to_csv(p,index=False)
        pd.DataFrame({'cation':['TEST'],'anion':['a'],'a':[999],'b':[999]}).to_csv(directory/'test.csv',index=False)
        if stage==3:
            extra=directory/'IL/cv2/fold1.csv';extra.parent.mkdir(parents=True)
            pd.DataFrame({'cation':['REPEAT'],'anion':['a'],'b':[999]}).to_csv(extra,index=False)
    pd.DataFrame(entries).to_csv(root/'task_catalog.csv',index=False)
    return root


def test_discovery_and_end_to_end(tmp_path,config):
    root=fixture_root(tmp_path)
    config['formulas'].update({'simulation/a.csv':'unconditioned','experiment/b.csv':'unconditioned',
                              'experiment/c.csv':'unconditioned'})
    config['stability'].update(bootstrap=2,permutation=2,cv_repeats=2,cv_permutation=2)
    nodes,inputs=graph.discover(root,config)
    assert len(nodes)==6
    assert all('test.csv' not in p and '/cv2/' not in p for p in inputs)
    inv,review,sig,units=graph.build_signatures(nodes,config)
    assert len(sig)==14
    assert not set(sig.cation)&{'TEST','REPEAT'}
    assert inv.excluded_reason.eq('excluded_non_il_identity').sum()==2
    out=tmp_path/'results';out.mkdir()
    graph.compute_graphs(nodes,sig,config,out,42,dpi=30)
    ee=pd.read_csv(out/'G_EE/n_shared.csv',index_col=0)
    assert ee.shape==(2,2)
    assert ee.iloc[0,0]==5
    se=pd.read_csv(out/'G_SE/n_shared.csv',index_col=0)
    assert se.shape==(2,2) and (se.to_numpy()==2).all()
    assert set(graph.METRICS) == EXPECTED_METRICS
    for name in ('G_EE', 'G_SE'):
        directory = out/name
        expected_files = {f'{m}.csv' for m in EXPECTED_METRICS} | {
            'n_shared.csv', 'n_observation_shared.csv', 'pairs.csv', 'confidence.csv', 'na_reasons.csv',
            'signature_overlap_matrix.csv', 'signature_overlap_heatmap.png',
        }
        assert {p.name for p in directory.iterdir()} == expected_files
        overlap = pd.read_csv(directory/'signature_overlap_matrix.csv', index_col=0)
        expected_index = ({'experiment/b', 'experiment/c'} if name == 'G_EE'
                          else {'simulation/a::a', 'simulation/a::b'})
        assert set(overlap.index) == expected_index
        assert set(overlap.columns) == {'experiment/b', 'experiment/c'}
        expected_overlap = ee.to_numpy() if name == 'G_EE' else se.to_numpy()
        assert np.array_equal(overlap.to_numpy(), expected_overlap)
        assert mpl_image.imread(directory/'signature_overlap_heatmap.png').size > 0
        pairs = pd.read_csv(directory/'pairs.csv')
        assert {c[:-7] for c in pairs if c.endswith('_status')} == EXPECTED_METRICS
        assert set(pairs.columns) & EXPECTED_METRICS == EXPECTED_METRICS
        assert not any(c.startswith('continuous_mi') for c in pairs)
        assert not any(c.startswith('binary_mi') for c in pairs)
        assert set(pd.read_csv(directory/'confidence.csv').metric) == EXPECTED_METRICS
        assert set(pd.read_csv(directory/'na_reasons.csv').metric) <= EXPECTED_METRICS
    knowledge = out/'knowledge_graphs'
    assert {p.name for p in knowledge.iterdir()} == {f'{metric}.png' for metric in EXPECTED_METRICS}
    assert all(mpl_image.imread(path).size > 0 for path in knowledge.iterdir())
    assert not (out/'G_SS').exists()
    cp=tmp_path/'config.json';cp.write_text(json.dumps(config))
    audit=tmp_path/'audit';graph.run('audit',root,audit,cp)
    assert json.loads((audit/'manifest.json').read_text())['status']=='complete'
    assert not (audit/'knowledge_graphs').exists()
    computed=tmp_path/'computed';graph.run('compute',root,computed,cp,dpi=30)
    manifest=json.loads((computed/'manifest.json').read_text())
    assert manifest['dpi']==30
    assert set(manifest['output_hashes']) == {
        str(path.relative_to(computed)) for path in computed.rglob('*')
        if path.is_file() and path.suffix in {'.csv', '.png'}
    }
    with pytest.raises(ValueError,match='nonempty'):
        graph.run('audit',root,audit,cp)


def test_discrete_mi_stability_and_degenerate_reporting(config):
    x=np.arange(30.);y=np.arange(30.)
    vals,_=graph.metric_values(x,y)
    settings=dict(config['stability'],bootstrap=3,permutation=2,cv_repeats=2,cv_permutation=2)
    rows=graph.confidence_rows(x,y,vals,(None,None),42,settings)
    mi=next(r for r in rows if r['metric']=='binary_i_over_h')
    assert vals['binary_i_over_h'] == pytest.approx(1)
    assert mi['status']=='insufficient_valid_bootstrap'
    assert mi['bootstrap_valid']==3 and mi['bootstrap_failed']==0
    assert mi['permutation_valid']==2
    assert next(r for r in rows if r['metric']=='multiclass_mi')['bootstrap_valid']==3
    assert {r['metric'] for r in rows} == EXPECTED_METRICS
    assert rows==graph.confidence_rows(x,y,vals,(None,None),42,settings)

    constant,_=graph.metric_values(x,np.ones(30))
    unavailable=graph.confidence_rows(x,np.ones(30),constant,(None,None),42,settings)
    assert next(r for r in unavailable if r['metric']=='binary_i_over_h')['status']=='metric_unavailable'


@pytest.mark.parametrize('n,bins,counts', [
    (29,None,None),(30,3,[10]*3),(79,3,[26,26,27]),
    (80,4,[20]*4),(149,4,[37,37,37,38]),(150,5,[30]*5),
])
def test_multiclass_sample_boundaries(n,bins,counts):
    # Uneven value spacing distinguishes quantile bins from equal-width bins.
    x=np.exp(np.linspace(0,5,n))
    values,states=graph.metric_values(x,x)
    if bins is None:
        assert np.isnan(values['multiclass_mi']) and states['multiclass_mi']=='insufficient_samples'
    else:
        assert np.bincount(graph.quantile_labels(x,bins)).tolist()==counts
        probabilities=np.array(counts)/n
        assert values['multiclass_mi']==pytest.approx(-np.sum(probabilities*np.log(probabilities)))


def test_binary_threshold_and_directed_normalization():
    x=np.arange(40.); y=np.r_[np.zeros(30),np.ones(10)]
    values,_=graph.metric_values(x,y)
    reverse,_=graph.metric_values(y,x)
    expected_mi=.5*np.log(4/3)+.25*np.log(2/3)+.25*np.log(2)
    target_entropy=-.75*np.log(.75)-.25*np.log(.25)
    assert values['binary_i_over_h']==pytest.approx(expected_mi/target_entropy)
    assert reverse['binary_i_over_h']==pytest.approx(expected_mi/np.log(2))
    threshold_values,_=graph.metric_values(x,y,(100.,None))
    assert threshold_values['binary_i_over_h']==0


def test_knowledge_graph_direction_significance_and_strength():
    nodes = [
        {'node_id': 's', 'stage': 2, 'excluded_reason': ''},
        {'node_id': 'e1', 'stage': 3, 'excluded_reason': ''},
        {'node_id': 'e2', 'stage': 3, 'excluded_reason': ''},
    ]
    labels = {'s': 'simulation/s', 'e1': 'experiment/e1', 'e2': 'experiment/e2'}
    columns = {
        'spearman': [-.5, -.5], 'distance_correlation': [.6, .6],
        'binary_i_over_h': [.2, .3], 'multiclass_mi': [.4, .4],
        'predictability_cv_nmae': [.25, 2.],
    }
    ee_pairs = pd.DataFrame({'source': ['e1', 'e2'], 'target': ['e2', 'e1'], **columns})
    se_pairs = pd.DataFrame({'source': ['s'], 'target': ['e1'],
                             **{metric: [values[0]] for metric, values in columns.items()}})
    confidence = pd.DataFrame([
        {'source': source, 'target': target, 'metric': metric,
         'permutation_p': .01 if source == 's' else .2}
        for source, target in [('e1', 'e2'), ('e2', 'e1'), ('s', 'e1')]
        for metric in EXPECTED_METRICS
    ])
    count = len(EXPECTED_METRICS)
    results = {
        'G_EE': {'pairs': ee_pairs, 'confidence': confidence.iloc[:2 * count]},
        'G_SE': {'pairs': se_pairs, 'confidence': confidence.iloc[2 * count:]},
    }
    spearman = graph.build_knowledge_graph('spearman', nodes, results, labels)
    assert not spearman.is_directed() and set(spearman.edges()) == {('s', 'e1'), ('e1', 'e2')}
    assert spearman['s']['e1']['significant']
    assert spearman['e1']['e2']['negative']
    directed = graph.build_knowledge_graph('binary_i_over_h', nodes, results, labels)
    assert directed.is_directed()
    assert set(directed.edges()) == {('s', 'e1'), ('e1', 'e2'), ('e2', 'e1')}
    positions = graph.shared_spring_layout(spearman, 42)
    repeated = graph.shared_spring_layout(spearman, 42)
    assert all(np.array_equal(positions[node], repeated[node]) for node in positions)
    widths = graph.edge_widths('predictability_cv_nmae', [
        ('a', 'b', {'value': .25}), ('a', 'c', {'value': 2.}),
    ])
    assert widths[0] > widths[1]


def test_solvation_transfer_uses_il_solute_identity(tmp_path, config):
    nodes = [
        {'node_id': 'sim', 'source_dataset': 'simulation/density.csv', 'target_property': 'x',
         'stage': 2, 'excluded_reason': '', 'formula': 'linear_temperature'},
        {'node_id': 'solvation', 'source_dataset': 'experiment/solvation.csv', 'target_property': 'x',
         'stage': 3, 'excluded_reason': '', 'formula': 'solute_linear_temperature'},
        {'node_id': 'transfer', 'source_dataset': 'experiment/transfer.csv', 'target_property': 'x',
         'stage': 3, 'excluded_reason': '', 'formula': 'solute_only'},
    ]
    provenance = {'signature_kind': 'solute_controlled', 'reference_mismatch': False,
                  'actual_reference': '{}', 'extrapolated': False}
    signatures = pd.DataFrame([
        {'node_id': 'sim', 'system': '["c", "a"]', 'signature': 3., **provenance},
        {'node_id': 'solvation', 'system': '["c", "a"]', 'signature': 1., **provenance},
        {'node_id': 'transfer', 'system': '["c", "a"]', 'signature': 2., **provenance},
    ])
    unit_provenance = provenance | {'signature_kind': 'single_raw'}
    units = pd.DataFrame([
        {'node_id': node, 'cation': 'c', 'anion': 'a', 'solute': solute,
         'signature': value, **unit_provenance}
        for node, values in [('solvation', [1., 2.]), ('transfer', [2., 4.])]
        for solute, value in zip(['s1', 's2'], values)
    ])
    local = dict(config)
    local['stability'] = dict(config['stability'], bootstrap=2, permutation=2,
                              cv_repeats=2, cv_permutation=2)
    out = tmp_path/'solute-pair';out.mkdir()
    graph.compute_graphs(nodes, signatures, local, out, 42, dpi=30, unit_signatures=units)
    counts = pd.read_csv(out/'G_EE/n_shared.csv', index_col=0)
    assert counts.loc['solvation', 'solvation'] == 1
    assert counts.loc['transfer', 'transfer'] == 1
    assert counts.loc['solvation', 'transfer'] == 2
    pairs = pd.read_csv(out/'G_EE/pairs.csv')
    pair = pairs.loc[(pairs.source == 'solvation') & (pairs.target == 'transfer')].iloc[0]
    assert pair.system_identity == 'cation_anion_solute'
    assert pair.n_observation_shared == 2
    assert len(json.loads(pair.shared_systems)) == 2
    se_pairs = pd.read_csv(out/'G_SE/pairs.csv')
    assert set(se_pairs.system_identity) == {'cation_anion'}


def test_nmae_is_oof_median_error_ratio(monkeypatch):
    class MedianPredictor:
        def fit(self, X, y):
            self.median = np.median(y)
        def predict(self, X):
            return np.full(len(X), self.median)
    monkeypatch.setattr(graph, 'make_pipeline', lambda *args: MedianPredictor())
    x = np.linspace(-3, 3, 25)
    value, reason = graph.cv_nmae(x, x**3, 5)
    assert reason == 'ok' and value == 1


def test_pending_signatures_preserve_observation_overlap(tmp_path, config):
    ids = ['s', 'e']
    nodes = []
    for stage, node_id, formula in [(2, ids[0], 'unconditioned'), (3, ids[1], 'pending_formula')]:
        nodes.append(dict(node_id=node_id, source_dataset=node_id, target_property='y', stage=stage,
                          system_type='il', excluded_reason='', formula=formula, condition_columns='temperature_K',
                          files=[], frame=pd.DataFrame({'cation':['c','c'], 'anion':['a','a'],
                                                        'temperature_K':[300.,320.], 'y':[1.,2.]})))
    _, _, sig, _ = graph.build_signatures(nodes, config)
    out=tmp_path/'graph';out.mkdir()
    graph.compute_graphs(nodes, sig, config, out, 42, dpi=30)
    assert pd.read_csv(out/'G_SE/n_shared.csv',index_col=0).iloc[0,0] == 0
    assert pd.read_csv(out/'G_SE/n_observation_shared.csv',index_col=0).iloc[0,0] == 1
    for metric in graph.METRICS:
        assert np.isnan(pd.read_csv(out/f'G_SE/{metric}.csv',index_col=0).iloc[0,0])
    pairs=pd.read_csv(out/'G_SE/pairs.csv')
    assert pairs.target_pending_formula.all()
