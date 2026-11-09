import json
import numpy as np
import pytest
import run


def setup_record(tmp_path,monkeypatch):
    out=tmp_path/'results';cache=tmp_path/'cache';folder=cache/'fixture'
    folder.mkdir(parents=True)
    monkeypatch.setattr(run,'OUT',out);monkeypatch.setattr(run,'CACHE',cache)
    y=np.array([0,1,2]);subject=np.array(['a']*3);session=np.array(['1']*3)
    np.savez(folder/'meta.npz',y=y,subject=subject,session=session,context=np.array(['Graz']*3),run=np.zeros(3),onset=np.arange(3))
    (folder/'manifest.json').write_text(json.dumps(dict(protocol='loso',fold='a',
        train_subjects=['b'],eval_subjects=['a'],train_windows=9,eval_windows=3)))
    p=np.broadcast_to(np.eye(3)[None]*.8+.2/3,(50,3,3)).copy()
    run.save_result('baseline','fixture',p,dict(classifier_seconds=0))
    return out,folder


def test_resume_rejects_changed_feature_manifest(tmp_path,monkeypatch):
    out,folder=setup_record(tmp_path,monkeypatch)
    assert run.validate_record('baseline','fixture')['eval_windows']==3
    with (folder/'manifest.json').open('a') as f:f.write(' ')
    with pytest.raises(RuntimeError,match='feature manifest'):
        run.validate_record('baseline','fixture')


def test_resume_rejects_inconsistent_metric_even_with_valid_prediction_digest(tmp_path,monkeypatch):
    out,folder=setup_record(tmp_path,monkeypatch)
    path=out/'baseline/fixture.json';record=json.loads(path.read_text());record['cell_mean']=.1
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError,match='metric'):
        run.validate_record('baseline','fixture')


def test_pipeline_overlaps_folds_but_drains_cv_before_test(tmp_path,monkeypatch):
    events=[]
    class Future:
        def __init__(self,name,tag):self.name,self.tag=name,tag
        def result(self):events.append(('cpu_done',self.tag,self.name))
    class Pool:
        def __init__(self,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,fn,name,tag):return Future(name,tag)
    monkeypatch.setattr(run,'OUT',tmp_path/'results');monkeypatch.setattr(run,'CACHE',tmp_path/'cache')
    monkeypatch.setattr(run,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(run,'as_completed',lambda jobs:iter(jobs))
    monkeypatch.setattr(run,'folds',lambda phase:iter([
        ('loso_a','loso','a',None,None),('loso_b','loso','b',None,None),
        ('later_last','later','last',None,None),('test_official','test','official',None,None)]))
    monkeypatch.setattr(run,'prepare',lambda tag,*args:events.append(('features',tag)))
    monkeypatch.setattr(run,'run_classical',lambda name,tag:events.append(('baseline',tag)))
    monkeypatch.setattr(run,'run_mlp',lambda tag:events.append(('mlp',tag)))
    monkeypatch.setattr(run,'freeze_selection',lambda:events.append(('freeze',)))
    monkeypatch.setattr('sys.argv',['run.py','--phase','all'])
    run.main()
    assert events.index(('features','loso_b'))<next(i for i,e in enumerate(events) if e[0]=='cpu_done')
    first_freeze=events.index(('freeze',))
    assert sum(e[0]=='cpu_done' for e in events[:first_freeze])==15
    assert first_freeze<events.index(('features','test_official'))
