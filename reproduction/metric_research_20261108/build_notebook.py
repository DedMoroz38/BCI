"""Private,budgeted Kaggle notebook; fixed public inputs and pinned author code."""
import base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nbf
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PROJECT=ROOT.parent

def main():
    files=[HERE/n for n in ('run_gpu.py','summarize.py','local_fusion.py','test_core.py','test_training.py','PROTOCOL.md')]
    files+=list((HERE/'src').glob('*.py'))
    source=ROOT/'ensemble_author_20261108'
    files+=list((source/'track2_bci/bci').rglob('*.py'))
    files += [p for p in source.glob('LICENSE*') if p.is_file()]
    original={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if source in p.parents and p.suffix=='.py'}
    buffer=io.BytesIO();manifest={}
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            data=p.read_bytes();name=str(p.relative_to(PROJECT));manifest[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
        receipt=json.dumps(dict(commit='934f423a73e1ed50d653345d27256ecfe50ab009',files=original),indent=2).encode()
        name='reproduction/ensemble_author_20261108/SOURCE_RECEIPT.json';z.writestr(name,receipt);manifest[name]=hashlib.sha256(receipt).hexdigest()
        z.writestr('metric-manifest.json',json.dumps(manifest,indent=2))
    payload=buffer.getvalue();digest=hashlib.sha256(payload).hexdigest()
    previous=nbf.read(ROOT/'single_attention_ablation/notebook/single_e2e.ipynb',as_version=4)
    bootstrap=previous.cells[1].source
    bootstrap=re.sub(r'payload=base64.b64decode\("[^"\n]+"\)',lambda m:'payload=base64.b64decode("'+base64.b64encode(payload).decode()+'")',bootstrap)
    bootstrap=re.sub(r'assert hashlib.sha256\(payload\).hexdigest\(\)=="[^"]+"','assert hashlib.sha256(payload).hexdigest()=="'+digest+'"',bootstrap)
    bootstrap=bootstrap.replace('single-manifest.json','metric-manifest.json').replace('experiment=repro/"single_attention_ablation"','experiment=repro/"metric_research_20261108"')
    bootstrap=bootstrap.replace('/tmp/single-env','/tmp/metric-env').replace('/tmp/single-mpl','/tmp/metric-mpl').replace('/kaggle/working/single-results','/kaggle/working/metric-results')
    bootstrap=bootstrap.replace('SINGLE_OUTPUT=str(out)','METRIC_OUTPUT=str(out)')
    bootstrap=re.sub(r'single_bundle_sha256="[^"]+"','metric_bundle_sha256="'+digest+'"',bootstrap)
    bootstrap=bootstrap.replace('Verified sources and data. SINGLE models,NO forest,full end-to-end gradients.','Verified public data and author ShallowNet. All10people,nested early stopping,75min budget.')
    bootstrap += '\nprint("CUDA DEVICES",torch.cuda.device_count(),[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],flush=True)\n'
    tests='''subprocess.run([python,"-m","pytest","-q","test_core.py","test_training.py","--tb=short"],cwd=experiment,env=env,check=True)
smoke="""import sys;sys.path.insert(0,'src')
import torch
from neural_training import aligned_pair,fit,CONFIGS
from bci.data import synthetic
torch.set_num_threads(2)
w=synthetic(n_subjects=3,multi_session=0,runs=2,per_run=6)
x,_,_=aligned_pair(w,w)
model,r=fit(torch.as_tensor(x,device='cuda:0'),w,CONFIGS['shallow16_subject'],0,'/tmp/metric-cuda-smoke','cuda:0',epochs=1)
assert r['parameters']>0
print('CUDA weighted training and checkpoint smoke passed',flush=True)
"""
subprocess.run([python,"-c",smoke],cwd=experiment,env=env,check=True)
'''
    execution='''from IPython.display import display,Image
figures={}
process=subprocess.Popen([python,"-u",str(experiment/"run_gpu.py"),"--minutes","75"],cwd=experiment,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
for line in process.stdout:
    print(line,end="",flush=True)
code=process.wait()
if code:raise RuntimeError(f"Metric research exited{code};checkpoints and partial predictions are saved")
'''
    nb=nbf.read(HERE/'notebook/metric_wave1.ipynb',as_version=4)
    nb.cells=[nbf.v4.new_markdown_cell('# PROTEUS metric research — wave1\n\nPrivate T4 experiment;no competition submission. All10public people:10LOSO folds and later sessions of2people.\n\nModels:authorShallow40,class loss;Shallow16,class loss;Shallow16,subject/class-balanced loss. AdamW.001,WD.01,dropout.5,causalEAprior5. Inner people-held-out selection,max30epochs,patience6;fresh refit for selected epoch count. Train/val curves saved.\n\nSeed0 all configurations first;seed1 compact models if budget permits. Two workers on separate GPUs;75min scheduling budget with checkpoint preservation. Partial model/seed sets are explicitly labeled.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_markdown_cell('## Data/split/loss/checkpoint and CUDA smoke tests'),nbf.v4.new_code_cell(tests),nbf.v4.new_markdown_cell('## Nested epoch selection and outer evaluation'),nbf.v4.new_code_cell(execution),nbf.v4.new_markdown_cell('## Completed-fit summary'),nbf.v4.new_code_cell('print((out/"REPORT.md").read_text())\nprint("Saved learning curves:",len(list(out.rglob("curves.png"))))')]
    nbf.validate(nb);nbf.write(nb,HERE/'notebook/metric_wave1.ipynb')
    meta=dict(id='den1shh/proteus-metric-wave1',title='PROTEUS Metric Wave1',code_file='metric_wave1.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=['den1shh/proteus-lag4-classifier-inputs','den1shh/proteus-lag4-classifier-source'],competition_sources=[],kernel_sources=[])
    (HERE/'notebook/kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (HERE/'notebook-build.json').write_text(json.dumps(dict(bundle_sha256=digest,source_manifest=manifest,notebook_sha256=hashlib.sha256((HERE/'notebook/metric_wave1.ipynb').read_bytes()).hexdigest()),indent=2))
    print('BUILT',len(manifest),'source files',len(payload),'bytes;privateT4;75min budget')

if __name__=='__main__':main()
