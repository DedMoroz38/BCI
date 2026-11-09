import json
import pytest
from run_single import check_runtime

def test_resume_rejects_library_changes(tmp_path):
    path=tmp_path/'runtime.json'
    a=dict(source_sha256='a',input_sha256={'train':'x'},versions={'torch':'1'})
    check_runtime(path,a)
    saved=path.read_bytes()
    b={**a,'versions':{'torch':'2'}}
    with pytest.raises(RuntimeError,match='versions'):check_runtime(path,b)
    assert saved==path.read_bytes()
