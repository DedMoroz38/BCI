import json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]

def test_isolated_smoke_all_configs_and_fit_roundtrip(tmp_path):
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/proteus-mpl')
    env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,'smoke.py','--device','cpu','--out',str(tmp_path)],cwd=HERE,env=env,
                          text=True,capture_output=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'ALL35 GRADIENT SMOKES PASSED' in result.stdout
    assert (tmp_path/'swad0.05/refit/complete.json').exists()

def test_generated_notebook_matches_bundle_and_private_gpu():
    import hashlib
    nb=json.loads((HERE/'notebook/regularization.ipynb').read_text())
    meta=json.loads((HERE/'notebook/kernel-metadata.json').read_text())
    build=json.loads((HERE/'notebook-build.json').read_text())
    assert meta['is_private'] and meta['enable_gpu'] and meta['kernel_sources']==[]
    assert hashlib.sha256((HERE/'notebook/regularization.ipynb').read_bytes()).hexdigest()==build['notebook_sha256']
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    assert '"240"' in code and 'smoke.py' in code and 'notebook-wait' not in code
