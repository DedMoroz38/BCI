from pathlib import Path
from run_attention import frozen_runtime

def test_runtime_rejects_changed_input_and_preserves_receipt(tmp_path):
    import pytest
    p=tmp_path/'runtime.json'
    frozen_runtime(p,dict(source_sha256='a',input_sha256={'train':'old'},device='cpu'))
    old=p.read_bytes()
    with pytest.raises(RuntimeError,match='provenance'):
        frozen_runtime(p,dict(source_sha256='a',input_sha256={'train':'changed'},device='cpu'))
    assert p.read_bytes()==old
    with pytest.raises(RuntimeError,match='provenance'):
        frozen_runtime(p,dict(source_sha256='different',input_sha256={'train':'old'},device='cpu'))
