import json
from pathlib import Path
import pytest
from run_itransformer import verify_cache,validate_runtime,partition_subjects
from training import digest
import numpy as np

def test_cache_tamper_rejected(tmp_path):
    p=tmp_path/'lag4_oas_band0.npy';np.save(p,np.eye(2))
    (tmp_path/'manifest.json').write_text('{}')
    (tmp_path/'lag4_oas_band0.complete.json').write_text(json.dumps({'sha256':digest(p)}))
    verify_cache(tmp_path)
    p.write_bytes(p.read_bytes()+b'x')
    with pytest.raises(RuntimeError,match='cache'):verify_cache(tmp_path)

def test_hardware_resume_change_rejected(tmp_path):
    path=tmp_path/'r.json'
    r=dict(source_sha256='a',input_sha256={},cache_sha256={},versions={},device='cuda',gpu='T4')
    validate_runtime(path,r)
    with pytest.raises(RuntimeError,match='device'):validate_runtime(path,{**r,'device':'cpu'})

def test_outer_person_cannot_enter_inner_selection():
    people=np.repeat(['a','b','c','d','e'],3)
    a,b=partition_subjects(people,np.array(['z']),'loso')
    assert not set(people[a])&set(people[b])
    with pytest.raises(RuntimeError,match='outer'):partition_subjects(people,np.array(['a']),'loso')
