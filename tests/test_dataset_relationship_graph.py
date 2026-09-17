import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_dataset_relationship_graph as graph


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


def test_vft_recovery_and_insufficient(config):
    refs=config['references'];t=np.linspace(280,400,15)
    y=2+400*(1/(t-150)-1/(refs['temperature_K']-150))
    fit=graph.fit_signature(frame(y,t,np.full(len(t),101.325)), 'vft', refs)
    assert fit['status']=='ok'
    assert fit['signature']==pytest.approx(2,abs=1e-5)
    assert json.loads(fit['parameters'])['T0']==pytest.approx(150,abs=1e-3)
    assert graph.fit_signature(frame(y[:3],t[:3],[100]*3),'vft',refs)['status']=='insufficient_vft_design'


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
    assert small['binary_mi']==pytest.approx(np.log(2))
    assert states['spearman']=='insufficient_samples'
    vals,states=graph.metric_values(np.ones(30),np.ones(30))
    assert vals['binary_mi']==0 and states['binary_i_over_h']=='zero_target_entropy'
    assert states['multiclass_mi']=='degenerate_quantile_bins'
    _,states=graph.metric_values(np.array([]),np.array([]))
    assert set(states.values())=={'no_shared_signatures'}
    first,_=graph.metric_values(np.arange(30.),np.arange(30.))
    assert first['binary_mi']==pytest.approx(np.log(2))
    assert graph.quantile_labels(np.ones(30),3) is None
    assert np.isnan(graph.ksg_mi(np.ones(30),np.ones(30)))


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
        (2,'simulation/qm','simulation/qm.csv','q1;q2','molecule',1)]:
        materialized=f'stage{stage}/{task.split("/")[-1]}'
        entries.append(dict(stage=stage,task_id=task,source_file=source,target_columns=targets,
                            system_type=system_type,materialized_path=materialized,repeats=repeats,condition_columns=''))
        if system_type!='il':continue
        directory=root/materialized;directory.mkdir(parents=True)
        names=['train.csv','valid.csv'] if stage==2 else [f'IL/cv1/fold{i}.csv' for i in range(1,6)]
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
    config['formulas'].update({'simulation/a.csv':'unconditioned','experiment/b.csv':'unconditioned'})
    config['stability'].update(bootstrap=2,permutation=2,cv_repeats=2,cv_permutation=2)
    nodes,inputs=graph.discover(root,config)
    assert len(nodes)==5
    assert all('test.csv' not in p and '/cv2/' not in p for p in inputs)
    inv,review,sig,units=graph.build_signatures(nodes,config)
    assert len(sig)==9
    assert not set(sig.cation)&{'TEST','REPEAT'}
    assert inv.excluded_reason.eq('excluded_non_il_identity').sum()==2
    out=tmp_path/'results';out.mkdir()
    graph.compute_graphs(nodes,sig,config,out,42)
    ee=pd.read_csv(out/'G_EE/n_shared.csv',index_col=0)
    assert ee.iloc[0,0]==5
    se=pd.read_csv(out/'G_SE/n_shared.csv',index_col=0)
    assert se.shape==(2,1) and (se.to_numpy()==2).all()
    assert not (out/'G_SS').exists()
    cp=tmp_path/'config.json';cp.write_text(json.dumps(config))
    audit=tmp_path/'audit';graph.run('audit',root,audit,cp)
    assert json.loads((audit/'manifest.json').read_text())['status']=='complete'
    with pytest.raises(ValueError,match='nonempty'):
        graph.run('audit',root,audit,cp)


def test_bootstrap_degenerate_reporting(config):
    x=np.arange(30.);y=x**2
    vals,_=graph.metric_values(x,y)
    settings=dict(config['stability'],bootstrap=3,permutation=2,cv_repeats=2,cv_permutation=2)
    rows=graph.confidence_rows(x,y,vals,(None,None),42,settings)
    mi=next(r for r in rows if r['metric']=='continuous_mi')
    assert mi['status']=='insufficient_valid_bootstrap' and mi['bootstrap_failed']==3
    assert rows==graph.confidence_rows(x,y,vals,(None,None),42,settings)


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
    graph.compute_graphs(nodes, sig, config, out, 42)
    assert pd.read_csv(out/'G_SE/n_shared.csv',index_col=0).iloc[0,0] == 0
    assert pd.read_csv(out/'G_SE/n_observation_shared.csv',index_col=0).iloc[0,0] == 1
    for metric in graph.METRICS:
        assert np.isnan(pd.read_csv(out/f'G_SE/{metric}.csv',index_col=0).iloc[0,0])
    pairs=pd.read_csv(out/'G_SE/pairs.csv')
    assert pairs.target_pending_formula.all()
