import json
import numpy as np
import pytest
import run

def test_folds_and_heldout_people():
    jobs=run.folds('cv')
    assert len(jobs)==9
    assert len(run.folds('test'))==1
    for tag,train,test in jobs[:8]:
        assert len(set(train.subject))==7
        assert len(set(test.subject))==1
        assert not set(train.subject)&set(test.subject)
    assert len(set(jobs[-1][2].subject))==1

def test_manifest_rejects_change(tmp_path):
    run.immutable_json(tmp_path/'m.json',{'code':'a'})
    run.immutable_json(tmp_path/'m.json',{'code':'a'})
    with pytest.raises(RuntimeError):
        run.immutable_json(tmp_path/'m.json',{'code':'b'})

def test_control_metadata_cannot_be_reordered():
    _,_,test=run.folds('cv')[0]
    values=run.metadata(test)
    run.check_metadata(test,values)
    values['onset']=values['onset'][::-1]
    with pytest.raises(AssertionError):run.check_metadata(test,values)

def test_stream_roundtrip_reset_and_causal_prefix(tmp_path):
    from bci.data import synthetic
    from bci.ensemble import fit_components,assemble
    from bci.stream import save_spec,load_spec,StreamModel
    w=synthetic(n_subjects=4,multi_session=1,runs=1,per_run=6)
    tr=w.select(w.subject!='sub-3');te=w.select(w.subject=='sub-3')
    parts=fit_components(tr,[{'balance':True,'routing':True}],n_trees=2,n_subject_trees=1)
    spec,model=assemble(parts,{'balance':True,'routing':True},tr.ch_names)
    p=run.probabilities(model,te)
    np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
    np.testing.assert_array_equal(p,run.probabilities(model,te))
    prefix=te.select(np.arange(len(te))<3)
    np.testing.assert_array_equal(p[:3],run.probabilities(model,prefix))
    save_spec(spec,tmp_path/'model.npz')
    np.testing.assert_array_equal(p,run.probabilities(StreamModel(load_spec(tmp_path/'model.npz')),te))
