"""Bundle pinned sources and verified control predictions into a private notebook."""
import base64,hashlib,io,json,re,shutil,zipfile
from pathlib import Path
import nbformat as nbf
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PROJECT=ROOT.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def reference():
    wave=ROOT/'metric_research_20261108';ref=HERE/'reference';manifest={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        manifest[str(target.relative_to(ref))]=sha(target)
    for folder in sorted((wave/'experiments/all10_baseline/results').iterdir()):
        if not folder.is_dir():continue
        for name in ('lag4','lag4_balance1'):
            record=json.loads((folder/(name+'.json')).read_text())
            assert sha(folder/(name+'.npz'))==record['prediction_sha256']
            copy(folder/(name+'.npz'),ref/'baselines'/folder.name/(name+'.npz'))
    assert len(manifest)==22
    (ref/'manifest.json').write_text(json.dumps(manifest,indent=2))

def main():
    reference()
    files=[HERE/n for n in ('run_gpu.py','runner_support.py','summarize.py','smoke.py','PROTOCOL.md')]
    files+=list((HERE/'src').glob('*.py'))+list((HERE/'reference/baselines').rglob('*.npz'))+[HERE/'reference/manifest.json']
    files += [HERE/'tests'/n for n in ('test_regularizers.py','test_training.py','test_pipeline.py')]
    files += list((HERE/'literature/vendor').glob('*'))
    author=ROOT/'ensemble_author_20261108';authorfiles=list((author/'track2_bci/bci').rglob('*.py'))
    files+=authorfiles+list(author.glob('LICENSE*'))
    receipt=json.dumps(dict(commit='934f423a73e1ed50d653345d27256ecfe50ab009',files={str(p.relative_to(author)):sha(p) for p in authorfiles}),indent=2).encode()
    buffer=io.BytesIO();manifest={}
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(p for p in files if p.is_file())):
            name=str(path.relative_to(PROJECT));data=path.read_bytes();manifest[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
        name='reproduction/ensemble_author_20261108/SOURCE_RECEIPT.json';manifest[name]=hashlib.sha256(receipt).hexdigest();z.writestr(name,receipt)
        z.writestr('regularization-manifest.json',json.dumps(manifest,indent=2))
    payload=buffer.getvalue();digest=hashlib.sha256(payload).hexdigest()
    old=nbf.read(ROOT/'metric_research_20261108/notebook/metric_wave1.ipynb',as_version=4)
    bootstrap=old.cells[1].source
    bootstrap=re.sub(r'payload=base64.b64decode\("[^"\n]+"\)',lambda m:'payload=base64.b64decode("'+base64.b64encode(payload).decode()+'")',bootstrap)
    bootstrap=re.sub(r'assert hashlib.sha256\(payload\).hexdigest\(\)=="[^"]+"','assert hashlib.sha256(payload).hexdigest()=="'+digest+'"',bootstrap)
    bootstrap=bootstrap.replace('metric-manifest.json','regularization-manifest.json').replace('experiment=repro/"metric_research_20261108"','experiment=repro/"regularization_research_20261108"')
    bootstrap=bootstrap.replace('/tmp/metric-env','/tmp/reg-env').replace('/tmp/metric-mpl','/tmp/reg-mpl').replace('/kaggle/working/metric-results','/kaggle/working/regularization-results')
    bootstrap=bootstrap.replace('METRIC_OUTPUT=str(out)','REG_OUTPUT=str(out),REG_CACHE="/tmp/proteus-regularization-cache"')
    bootstrap=re.sub(r'metric_bundle_sha256="[^"]+"','regularization_bundle_sha256="'+digest+'"',bootstrap)
    bootstrap=bootstrap.replace('All10people,nested early stopping,75min budget.','All10people,35regularizers,unchanged NLL stopping,240min cap.')
    tests='''subprocess.run([python,"-m","pytest","-q","tests/test_regularizers.py","tests/test_training.py","tests/test_pipeline.py","--tb=short"],cwd=experiment,env=env,check=True)
subprocess.run([python,"-u","smoke.py","--device","cuda:0"],cwd=experiment,env=env,check=True)
'''
    execution='''process=subprocess.Popen([python,"-u","run_gpu.py","--minutes","240"],cwd=experiment,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
for line in process.stdout:print(line,end="",flush=True)
code=process.wait()
if code:raise RuntimeError(f"Regularization study exited{code};partial checkpoints preserved")
'''
    nb=nbf.read(HERE/'notebook/regularization.ipynb',as_version=4)
    nb.cells=[nbf.v4.new_markdown_cell('# PROTEUS regularization wave2\n\n35 variants,2 seeds,11 outer folds (770 fits). Simple changes first,then six fixed combinations. Label smoothing;input/feature augmentations;window/EMA/SWAD averaging.\n\n**Unchanged:** inner cell-NLL epoch selection,Shallow16,class-balanced loss,AdamW.001,WD.01,dropout.5,causal alignment prior5. No pretrained models.\n\nOnly regularizers run;no separate controls. Historical results are reused for comparison. Clean held-out evaluation;training-only BN recalibration. Two T4workers,240minute wall cap;partial results labeled. No competition submission.'),
        nbf.v4.new_code_cell(bootstrap),nbf.v4.new_markdown_cell('## CPU regression tests and all35 CUDA gradient smokes'),nbf.v4.new_code_cell(tests),
        nbf.v4.new_markdown_cell('## Simple-first grid;NLL selection and fresh outer refit'),nbf.v4.new_code_cell(execution),
        nbf.v4.new_markdown_cell('## Complete two-seed configurations only'),nbf.v4.new_code_cell('print((out/"REPORT.md").read_text())\nprint("Learning curves:",len(list(out.rglob("curves.png"))))')]
    nbf.validate(nb);nbf.write(nb,HERE/'notebook/regularization.ipynb')
    meta=dict(id='den1shh/proteus-regularization-wave2',title='PROTEUS Regularization Wave2',code_file='regularization.ipynb',language='python',kernel_type='notebook',
        is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',
        dataset_sources=['den1shh/proteus-lag4-classifier-inputs','den1shh/proteus-lag4-classifier-source'],competition_sources=[],kernel_sources=[])
    (HERE/'notebook/kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (HERE/'notebook-build.json').write_text(json.dumps(dict(bundle_sha256=digest,source_manifest=manifest,notebook_sha256=sha(HERE/'notebook/regularization.ipynb')),indent=2))
    print('BUILT',len(manifest),'files',len(payload),'bytes;privateT4,240min cap',flush=True)

if __name__=='__main__':main()
