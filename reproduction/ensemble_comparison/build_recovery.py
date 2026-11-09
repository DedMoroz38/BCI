"""Private recovery notebook attaches immutable previous notebook outputs."""
import base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from build_notebook import HERE,bundle

def main():
    payload,manifest=bundle(recovery=True);digest=hashlib.sha256(payload).hexdigest()
    nb=nbf.read(HERE/'notebook/author_ensemble_gpu.ipynb',as_version=4)
    bootstrap=nb.cells[1].source
    bootstrap=re.sub(r'payload=base64.b64decode\("[^"\n]+"\)',lambda m:'payload=base64.b64decode("'+base64.b64encode(payload).decode()+'")',bootstrap)
    bootstrap=re.sub(r'assert hashlib.sha256\(payload\).hexdigest\(\)=="[^"]+"','assert hashlib.sha256(payload).hexdigest()=="'+digest+'"',bootstrap)
    bootstrap=re.sub(r'ensemble_bundle_sha256="[^"]+"','ensemble_bundle_sha256="'+digest+'"',bootstrap)
    # Previous kernel output contains an earlier source tree; select the dedicated dataset manifests only.
    bootstrap=bootstrap.replace('sources=list(pathlib.Path("/kaggle/input").rglob("source-integrity.json"));assert len(sources)==1', 'sources=[p for p in pathlib.Path("/kaggle/input").rglob("source-integrity.json") if "proteus-lag4-classifier-source" in str(p)];assert len(sources)==1')
    bootstrap=bootstrap.replace('inputs=list(pathlib.Path("/kaggle/input").rglob("input-integrity.json"));assert len(inputs)==1', 'inputs=[p for p in pathlib.Path("/kaggle/input").rglob("input-integrity.json") if "proteus-lag4-classifier-inputs" in str(p)];assert len(inputs)==1')
    bootstrap += '\nprint("CUDA DEVICES",torch.cuda.device_count(),flush=True)\n'
    checkpoint=json.loads((HERE/'checkpoint-input/checkpoint-integrity.json').read_text())
    checkpoint_receipt_sha=hashlib.sha256((HERE/'checkpoint-input/checkpoint-integrity.json').read_bytes()).hexdigest()
    bootstrap += f'''\nreceipts=list(pathlib.Path('/kaggle/input').rglob('checkpoint-integrity.json'));assert len(receipts)==1
assert hashlib.sha256(receipts[0].read_bytes()).hexdigest()=="{checkpoint_receipt_sha}"
checkpoint_receipt=json.loads(receipts[0].read_text())
parents=list(receipts[0].parent.rglob('ensemble-results/manifest.json'));assert len(parents)==1
parent_dir=parents[0].parent
for name,digest in checkpoint_receipt['files'].items():
    rel=pathlib.PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
    assert hashlib.sha256((parent_dir/name).read_bytes()).hexdigest()==digest,name
env['ENSEMBLE_PARENT']=str(parent_dir)
env['ENSEMBLE_PARENT_MANIFEST_SHA256']="{checkpoint['parent_manifest_sha256']}"
print('Verified pinned parent checkpoints for6folds',flush=True)
'''
    nb.cells[0].source='# PROTEUS author ensemble vs Lag4: checkpoint recovery\n\nResume verified v2 CUDA checkpoints; compute missing evaluations and fits only. Same models,folds and hyperparameters. Exact serialized parameters and labels; FP32 probability replay tolerance1e-4. Full parent/result lineage saved.'
    nb.cells[1].source=bootstrap
    nb.cells[3].source='subprocess.run([python,"-m","pytest","-q","test_runner.py","test_analysis.py","test_replay.py","--tb=short"],cwd=experiment,env=env,check=True)'
    nb.cells[5].source=nb.cells[5].source.replace('str(experiment/"run.py"),"--jobs","2"','str(experiment/"resume.py"),"--jobs","2"')
    folder=HERE/'recovery-notebook';folder.mkdir(exist_ok=True)
    nbf.validate(nb);nbf.write(nb,folder/'recovery.ipynb')
    meta=json.loads((HERE/'notebook/kernel-metadata.json').read_text())
    meta.update(id='den1shh/proteus-author-ensemble-recovery',title='PROTEUS Author Ensemble Recovery',code_file='recovery.ipynb',kernel_sources=[])
    meta['dataset_sources'].append('den1shh/proteus-author-ensemble-checkpoints')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (HERE/'recovery-build.json').write_text(json.dumps(dict(bundle_sha256=digest,files=manifest,notebook_sha256=hashlib.sha256((folder/'recovery.ipynb').read_bytes()).hexdigest()),indent=2))
    print('RECOVERY BUNDLE',len(manifest),'files',len(payload),'bytes;private T4;parent kernel v2')

if __name__=='__main__':main()
