import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import numpy as np
import pytest
import runner_support as s
from common import inner_split
from configurations import CONFIGS
from bci.data import synthetic

def test_grid_starts_regularizers_without_any_separate_controls():
    tags=[f'loso_P{i:02}' for i in range(1,11)]+['later_last']
    jobs=s.schedule(tags)
    assert len(jobs)==770
    assert all(j[0] not in ('control','bn_refresh') for j in jobs)
    assert [j[2] for j in jobs[:11]]==tags
    assert [j[1] for j in jobs[:22]]==[0]*11+[1]*11
    assert all(next(c['stage'] for c in CONFIGS if c['name']==j[0])<2 for j in jobs[:29*22])

def test_outer_and_inner_people_and_later_sessions():
    w=synthetic(n_subjects=4,multi_session=2,runs=2,per_run=6)
    tags=s.fold_tags(w);assert len(tags)==5
    for tag in tags:
        tr,te=s.split_outer(w,tag)
        assert not set(tr.stream)&set(te.stream)
        if tag.startswith('loso_'):assert not set(tr.subject)&set(te.subject)
        else:assert len(set(te.subject))==2
        a,b=inner_split(tr.subject);assert not set(tr.subject[a])&set(tr.subject[b])

def test_metrics_and_fixed_mixture_are_causal():
    from metrics import score,balanced_mix
    y=np.tile(np.arange(3),4);meta=dict(y=y,subject=np.array(['a']*6+['b']*6),session=np.array(['1']*12),context=np.array(['x']*12))
    p=np.eye(3)[y]*.8+.2/3
    assert score(meta,p)['cell_mean']==pytest.approx(1.)
    z=np.log(p);q=balanced_mix(meta,p,[z,z])
    first={k:v[:3] for k,v in meta.items()}
    np.testing.assert_allclose(q[:3],balanced_mix(first,p[:3],[z[:3],z[:3]]))

def test_existing_artifact_corruption_is_rejected_before_resume(tmp_path):
    import json
    from common import sha
    record={}
    for name,key in [('prediction.npz','prediction_sha256'),('model.pt','model_sha256'),('alignment.npz','alignment_sha256'),('logits.npy','logits_sha256')]:
        (tmp_path/name).write_bytes(name.encode());record[key]=sha(tmp_path/name)
    path=tmp_path/'prediction.json';path.write_text(json.dumps(record))
    s.validate_existing(path)
    (tmp_path/'model.pt').write_bytes(b'corruption')
    with pytest.raises(AssertionError):s.validate_existing(path)

def test_replay_drift_is_recorded_without_rejecting_independent_training():
    a=np.array([[.50001,.49999,0.]])
    same=s.replay_diagnostics(a,a);assert same['label_disagreements']==0
    drift=s.replay_diagnostics(a,a[:,[1,0,2]])
    assert drift['label_disagreements']==1 and drift['max_probability_delta']>0
    with pytest.raises(AssertionError):s.replay_diagnostics(a,np.full_like(a,np.nan))

def test_single_failed_model_does_not_abort_other_jobs(tmp_path,monkeypatch):
    import run_gpu as run
    monkeypatch.setattr(run,'OUT',tmp_path)
    def fail(args):raise RuntimeError('one experimental model failed')
    monkeypatch.setattr(run,'worker_impl',fail)
    r=run.worker(('noise0.03',0,'loso_P01',0))
    assert r['status']=='failed'
    assert (tmp_path/'models/loso_P01/noise0.03/seed0/failure.json').exists()
    monkeypatch.setattr(run,'worker_impl',lambda args:dict(status='complete'))
    assert run.worker(('noise0.03',0,'loso_P02',0))['status']=='complete'
