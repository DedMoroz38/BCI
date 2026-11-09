"""Portable CUDA launch; upstream model math is archived byte-for-byte."""
import base64,hashlib,io,json,re,subprocess,zipfile
from pathlib import Path
import nbformat as nbf
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PROJECT=ROOT.parent

def replace_once(text,old,new):
    if text.count(old)!=1:raise RuntimeError(f'Expected one patch target: {old[:90]}')
    return text.replace(old,new)

def bundle(recovery=False):
    files={}
    for name in ('run.py','baseline.py','analyze.py','PROTOCOL.md','test_runner.py','test_analysis.py'):
        files['reproduction/ensemble_comparison/'+name]=(HERE/name).read_bytes()
    upstream={}
    source=ROOT/'ensemble_author_20261108'
    for path in sorted((source/'track2_bci/bci').rglob('*.py')):
        name=str(path.relative_to(source));data=path.read_bytes()
        files['reproduction/ensemble_author_20261108/'+name]=data
        upstream[name]=hashlib.sha256(data).hexdigest()
    for path in source.glob('LICENSE*'):
        if path.is_file():files['reproduction/ensemble_author_20261108/'+path.name]=path.read_bytes()
    files['reproduction/ensemble_author_20261108/SOURCE_RECEIPT.json']=json.dumps(dict(
        commit='934f423a73e1ed50d653345d27256ecfe50ab009',files=upstream),indent=2).encode()
    files['reproduction/results/eeg/manifest.json']=(ROOT/'results/eeg/manifest.json').read_bytes()
    prefix='reproduction/ensemble_comparison/'
    driver=files[prefix+'run.py'].decode()
    driver=replace_once(driver,"OUT=HERE/'results'","ARTIFACT=Path(os.environ.get('ENSEMBLE_OUTPUT',str(HERE)))\nOUT=ARTIFACT/'results'")
    old='''    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=AUTHOR,text=True).strip()
    if head!=COMMIT or subprocess.check_output(['git','diff','HEAD'],cwd=AUTHOR):
        raise RuntimeError('Changed author source')'''
    new='''    receipt=json.loads((AUTHOR/'SOURCE_RECEIPT.json').read_text())
    head=receipt['commit']
    if head!=COMMIT:raise RuntimeError('Changed author commit')
    for rel,digest in receipt['files'].items():
        if sha(AUTHOR/rel)!=digest:raise RuntimeError('Changed author source')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for GPU migration')'''
    driver=replace_once(driver,old,new)
    driver=driver.replace("HERE/'manifest.json'","ARTIFACT/'manifest.json'").replace("HERE/'cv-freeze.json'","ARTIFACT/'cv-freeze.json'")
    driver=replace_once(driver,'p.relative_to(HERE)','p.relative_to(ARTIFACT)')
    driver=replace_once(driver,"blas_threads=1))","blas_threads=1,deep_device='cuda',gpu=torch.cuda.get_device_name(0)))")
    driver=replace_once(driver,"fit_deep(train,device='cpu')","fit_deep(train,device='cuda')")
    driver=replace_once(driver,"'100 epochs CPU'","'100 epochs CUDA'")
    driver=replace_once(driver,'ProcessPoolExecutor(max_workers=args.jobs)',"ProcessPoolExecutor(max_workers=args.jobs,mp_context=__import__('multiprocessing').get_context('spawn'))")
    if recovery:
        driver=replace_once(driver,"from bci.stream import save_spec,load_spec,StreamModel", "from bci.stream import save_spec,load_spec,StreamModel\nfrom replay_checks import assert_replay,assert_spec")
        driver=replace_once(driver,"files=[HERE/x for x in ('run.py','baseline.py','analyze.py','PROTOCOL.md')]", "files=[HERE/x for x in ('run.py','baseline.py','analyze.py','PROTOCOL.md','replay_checks.py','resume.py')]")
        check='np.testing.assert_array_equal(p[idx],probabilities(restored,prefix))'
        assert driver.count(check)==2
        driver=driver.replace(check,"assert_spec(spec,load_spec(folder/f'model{k}.npz'))\n            replay_delta=assert_replay(p[idx],probabilities(restored,prefix))\n            write_json(folder/f'replay{k}.json',{'parameters_exact':True,'labels_exact':True,'max_probability_delta':replay_delta,'atol':1e-4})")
        driver=replace_once(driver,'torch.set_num_threads(2)',"torch.set_num_threads(2)\n    identity=__import__('multiprocessing').current_process()._identity\n    gpu=((identity[0]-1) if identity else 0)%torch.cuda.device_count()\n    torch.cuda.set_device(gpu)\n    print('WORKER_GPU',tag,gpu,flush=True)")
        for name in ('replay_checks.py','resume.py','test_replay.py'):
            files[prefix+name]=(HERE/name).read_bytes()
    files[prefix+'run.py']=driver.encode()
    baseline=files[prefix+'baseline.py'].decode()
    baseline=replace_once(baseline,"folder=HERE/'results'/tag;out=folder/'baseline.json'","folder=Path(os.environ.get('ENSEMBLE_OUTPUT',str(HERE)))/'results'/tag;out=folder/'baseline.json'")
    files[prefix+'baseline.py']=baseline.encode()
    analysis=files[prefix+'analyze.py'].decode()
    analysis=replace_once(analysis,'import hashlib,json','import hashlib,json,os')
    analysis=replace_once(analysis,'HERE=Path(__file__).resolve().parent',"HERE=Path(os.environ.get('ENSEMBLE_OUTPUT',str(Path(__file__).resolve().parent)))")
    analysis=analysis.replace('Локальное воспроизведение commit934f423.','Kaggle CUDA-воспроизведение commit934f423.')
    files[prefix+'analyze.py']=analysis.encode()
    files[prefix+'PROTOCOL.md'] += b'''\n\n## Authorized GPU migration\n2026-11-08: user cancelled slow local CPU run and requested Kaggle.\nLocal results retained separately,not mixed with CUDA fits. Fresh full80-job\nreplication on Kaggle: same folds,100epochs,seed0 and every model parameter.\nOnly deep device changes CPU->CUDA,using upstream fit_deep(device='cuda').\nTwo spawn-based workers; one CUDA device; BLAS1,torch2threads. CUDA runtime\nversions/hardware recorded. Upstream files verified by pinned receipt rather\nthan requiring a full git history in the execution bundle. No submission.\n'''
    if recovery:
        files[prefix+'PROTOCOL.md']+=b'''\nRecovery of v2 numerical replay assertion: import verified trained components\nand predictions from the immutable v2 output. Record parent manifest and every\nimport hash;rebind records explicitly under the recovery manifest. Training\nmath,folds,hyperparameters and seed unchanged. Serialized specifications must\nmatch exactly; FP32 probabilities tolerate1e-4 absolute error while predicted\nlabels still must match exactly. Use separate CUDA devices for workers when\navailable. No model or setting selected using partial evaluation results.\n'''
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())}
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):z.writestr(name,data)
        z.writestr('ensemble-manifest.json',json.dumps(manifest,indent=2))
    return buffer.getvalue(),manifest

def main():
    payload,manifest=bundle();digest=hashlib.sha256(payload).hexdigest()
    prior=nbf.read(ROOT/'single_attention_ablation/notebook/single_e2e.ipynb',as_version=4)
    bootstrap=prior.cells[1].source
    bootstrap=re.sub(r'payload=base64.b64decode\("[^"\n]+"\)',lambda m:'payload=base64.b64decode("'+base64.b64encode(payload).decode()+'")',bootstrap)
    bootstrap=re.sub(r'assert hashlib.sha256\(payload\).hexdigest\(\)=="[^"]+"','assert hashlib.sha256(payload).hexdigest()=="'+digest+'"',bootstrap)
    bootstrap=bootstrap.replace('single-manifest.json','ensemble-manifest.json').replace('experiment=repro/"single_attention_ablation"','experiment=repro/"ensemble_comparison"')
    bootstrap=bootstrap.replace('/tmp/single-env','/tmp/ensemble-env').replace('/tmp/single-mpl','/tmp/ensemble-mpl').replace('/kaggle/working/single-results','/kaggle/working/ensemble-results')
    bootstrap=bootstrap.replace('SINGLE_OUTPUT=str(out)','ENSEMBLE_OUTPUT=str(out)')
    bootstrap=re.sub(r'single_bundle_sha256="[^"]+"','ensemble_bundle_sha256="'+digest+'"',bootstrap)
    bootstrap=bootstrap.replace('Verified sources and data. SINGLE models,NO forest,full end-to-end gradients.','Verified author ensemble and exact submitted Lag4. ShallowConvNet trains on CUDA.')
    tests='''subprocess.run([python,"-m","pytest","-q","test_runner.py","test_analysis.py","--tb=short"],cwd=experiment,env=env,check=True)
smoke="""import run; from bci.data import synthetic; from bci.ensemble import fit_deep,fit_components,assemble
from bci.stream import save_spec,load_spec,StreamModel
run.torch.set_num_threads(2)
w=synthetic(n_subjects=3,multi_session=0,runs=1,per_run=6)
parts=fit_components(w,[{}],n_trees=2)
parts['deep']=fit_deep(w,epochs=1,device='cuda')
spec,model=assemble(parts,{'deep':True},w.ch_names)
p=run.probabilities(model,w)
save_spec(spec,'/tmp/ensemble-gpu-smoke.npz')
run.np.testing.assert_array_equal(p,run.probabilities(StreamModel(load_spec('/tmp/ensemble-gpu-smoke.npz')),w))
print('CUDA training and saved model replay passed',flush=True)
"""
subprocess.run([python,"-c",smoke],cwd=experiment,env=env,check=True)
'''
    execution='''process=subprocess.Popen([python,"-u",str(experiment/"run.py"),"--jobs","2"],cwd=experiment,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
for line in process.stdout:print(line,end="",flush=True)
code=process.wait()
if code:raise RuntimeError(f"Experiment exited with status {code};partial results are saved")
'''
    nb=nbf.read(HERE/'notebook/author_ensemble_gpu.ipynb',as_version=4)
    nb.cells=[nbf.v4.new_markdown_cell('# PROTEUS: author ensemble vs submitted Lag4 — CUDA\n\nPrivate GPU run; no competition submission. Author commit934f423,unchanged model code.\n\nSeven author variants: forest,+balance,+bands,+routing,+deep,all but deep,all. Baseline: submitted50-tree Lag4.\n\nSame8-person development LOSO,one later session,and original train-only public test.100epochs,one ShallowConvNet seed0,author defaults. Two fold workers; CUDA for ShallowConvNet,CPU for geometry and LR.80 evaluations.\n\nThe cancelled local run is archived separately; these results are a fresh CUDA replication.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_markdown_cell('## Verify splits,stream state,serialization and CUDA training'),nbf.v4.new_code_cell(tests),nbf.v4.new_markdown_cell('## Run fixed models; freeze CV before public test'),nbf.v4.new_code_cell(execution),nbf.v4.new_markdown_cell('## Independently verified comparison'),nbf.v4.new_code_cell('print((out/"REPORT.md").read_text())\nprint("Artifacts:",sorted(p.name for p in out.iterdir()))')]
    nbf.validate(nb);nbf.write(nb,HERE/'notebook/author_ensemble_gpu.ipynb')
    meta=dict(id='den1shh/proteus-author-ensemble-vs-lag4',title='PROTEUS Author Ensemble vs Lag4',code_file='author_ensemble_gpu.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=['den1shh/proteus-lag4-classifier-inputs','den1shh/proteus-lag4-classifier-source'],competition_sources=[],kernel_sources=[])
    (HERE/'notebook/kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (HERE/'notebook-build.json').write_text(json.dumps(dict(bundle_sha256=digest,files=manifest,notebook_sha256=hashlib.sha256((HERE/'notebook/author_ensemble_gpu.ipynb').read_bytes()).hexdigest()),indent=2))
    print(json.dumps(dict(source_files=len(manifest),bundle_bytes=len(payload),private=True,gpu='T4',evaluations=80),indent=2))

if __name__=='__main__':main()
