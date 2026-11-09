import ast,hashlib,io,json,zipfile
import build_notebook as build

def test_gpu_bundle_preserves_upstream_and_uses_cuda():
    payload,manifest=build.bundle()
    z=zipfile.ZipFile(io.BytesIO(payload))
    for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    prefix='reproduction/ensemble_comparison/'
    driver=z.read(prefix+'run.py').decode()
    ast.parse(driver)
    assert "fit_deep(train,device='cuda')" in driver
    assert "fit_deep(train,device='cpu')" not in driver
    assert "get_context('spawn')" in driver
    assert 'p.relative_to(ARTIFACT)' in driver
    assert 'p.relative_to(HERE)' not in driver
    assert 'git\', \'rev-parse' not in driver
    upstream='reproduction/ensemble_author_20261108/track2_bci/bci/ensemble.py'
    assert z.read(upstream)==(build.ROOT/'ensemble_author_20261108/track2_bci/bci/ensemble.py').read_bytes()
    assert b'80' in z.read(prefix+'analyze.py')
    for name in ('baseline.py','test_runner.py','test_analysis.py'):
        ast.parse(z.read(prefix+name).decode())

def test_recovery_driver_keeps_strict_parameters_and_decisions():
    payload,_=build.bundle(recovery=True)
    z=zipfile.ZipFile(io.BytesIO(payload))
    prefix='reproduction/ensemble_comparison/'
    for name in ('run.py','resume.py','replay_checks.py'):
        ast.parse(z.read(prefix+name).decode())
    driver=z.read(prefix+'run.py').decode()
    assert driver.count('assert_spec(spec,load_spec(')==2
    assert driver.count('assert_replay(p[idx],probabilities(restored,prefix))')==2
    assert driver.count('np.testing.assert_array_equal(p[idx].argmax(1),restored.predict(')==2
    assert "fit_deep(train,device='cuda')" in driver
    resume=z.read(prefix+'resume.py').decode()
    assert "for key in ('data','cache','references','versions','variants')" in resume
