"""Package the exact experiment source and baseline receipts into a private notebook."""
from pathlib import Path
import argparse,base64,hashlib,io,json,subprocess,zipfile
import nbformat as nbf

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PROJECT=ROOT.parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-source',action='store_true');args=parser.parse_args()
    files=[]
    exp=ROOT/'classifier_ablation'
    files += list(exp.glob('*.py'))+list(exp.glob('*.md'))+list((exp/'literature').glob('*.md'))
    files += list((ROOT/'Brain-Decoding/track2_bci/bci').rglob('*.py'))
    files += [ROOT/'temporal_spd_research'/n for n in ['temporal_features.py','temporal_cache.py']]
    files += [ROOT/'2026-competition/benchmark_utils'/n for n in ['__init__.py','metrics.py']]
    files += [ROOT/'submission_lag4/package'/n for n in ['weights.npz','model_info.json']]
    for name in ['LICENSE','LICENSE.txt','LICENSE.md']:
        if (ROOT/'Brain-Decoding'/name).exists():files.append(ROOT/'Brain-Decoding'/name)
    control=ROOT/'temporal_spd_research/results/lag4_oas'
    files += list(control.glob('forest20_*.json'))+list(control.glob('forest20_*_predictions.npz'))
    buffer=io.BytesIO();hashes={}
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(set(files)):
            rel=str(file.relative_to(PROJECT));data=file.read_bytes()
            z.writestr(rel,data);hashes[rel]=hashlib.sha256(data).hexdigest()
        z.writestr('source-integrity.json',json.dumps(hashes,indent=2))
    code_input=HERE/'code-input';code_input.mkdir(exist_ok=True)
    payload=buffer.getvalue()
    if (code_input/'classifier-source.zip').exists() and not args.refresh_source:
        payload=(code_input/'classifier-source.zip').read_bytes()
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            hashes=json.loads(z.read('source-integrity.json'))
    (code_input/'classifier-source.zip').write_bytes(payload)
    (code_input/'dataset-metadata.json').write_text(json.dumps(dict(
        id='den1shh/proteus-lag4-classifier-source',title='PROTEUS Lag4 Classifier Source',
        licenses=[dict(name='other')],description='Private reproducibility bundle: experiment source, original BCI source with its license, and baseline prediction receipts. See the included licenses and protocol.'),indent=2))
    notebook=HERE/'notebook/classifier_ablation.ipynb'
    nb=nbf.read(notebook,as_version=4)
    bootstrap=f'''import os, sys, json, base64, hashlib, zipfile, io, pathlib, subprocess, importlib.metadata
os.environ.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
import torch
assert torch.cuda.is_available(), "Enable a CUDA GPU for the requested MLP comparison"
print("GPU", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],flush=True)
project=pathlib.Path("/kaggle/working/riemann-project")
project.mkdir(parents=True,exist_ok=True)
sources=list(pathlib.Path("/kaggle/input").rglob("source-integrity.json"))
assert len(sources)==1, sources
source_manifest=sources[0].read_bytes()
assert hashlib.sha256(source_manifest).hexdigest()=="{hashlib.sha256(json.dumps(hashes,indent=2).encode()).hexdigest()}"
for rel,digest in json.loads(source_manifest).items():
    path=pathlib.PurePosixPath(rel)
    assert not path.is_absolute() and ".." not in path.parts
    data=(sources[0].parent/rel).read_bytes()
    assert hashlib.sha256(data).hexdigest()==digest,rel
    dest=project/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
(project/"source-integrity.json").write_bytes(source_manifest)
matches=list(pathlib.Path("/kaggle/input").rglob("input-integrity.json"))
assert len(matches)==1, matches
inputs=matches[0].parent
manifest=json.loads(matches[0].read_text())
for name,digest in manifest["files_sha256"].items():
    h=hashlib.sha256()
    with (inputs/name).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    assert h.hexdigest()==digest,name
repro=project/"reproduction"
(repro/"data").mkdir(exist_ok=True)
for split in ["train","val","test"]:
    link=repro/"data"/f"proteus_{{split}}.npz"
    if not link.exists():link.symlink_to(inputs/link.name)
cache=repro/"temporal_spd_research/cache"
if not cache.exists():cache.symlink_to(inputs,target_is_directory=True)
envdir=pathlib.Path("/tmp/classifier-env")
subprocess.run([sys.executable,"-m","venv","--without-pip","--system-site-packages",str(envdir)],check=True)
python=str(envdir/"bin/python")
pins=["numpy==2.5.4","scipy==1.18.1","scikit-learn==1.9.1","pyriemann==0.12","pandas==3.1.0","threadpoolctl==3.7.0"]
subprocess.run([sys.executable,"-m","pip","--python",python,"install","--disable-pip-version-check","--quiet",*pins],check=True)
subprocess.run([python,"-c","import sys,torch,importlib.metadata as m;print('Numerical runtime',sys.version,{{k:m.version(k) for k in ['numpy','scipy','scikit-learn','pyriemann','pandas','torch']}});assert torch.cuda.is_available()"],check=True)
out=pathlib.Path("/kaggle/working/classifier-results");out.mkdir(exist_ok=True)
env=dict(os.environ,CLASSIFIER_OUTPUT=str(out),CLASSIFIER_CACHE="/tmp/classifier-features",PYTHONUNBUFFERED="1",MPLCONFIGDIR="/tmp/matplotlib-classifier")
runtime=dict(gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],torch=torch.__version__,python=sys.version,source_manifest_sha256=hashlib.sha256(source_manifest).hexdigest(),input_manifest=manifest)
(out/"runtime.json").write_text(json.dumps(runtime,indent=2))
print("Verified all source/input hashes. Original data and submitted model are attached.",flush=True)
'''
    tests='''experiment=repro/"classifier_ablation"
test_files=["test_classifiers.py","test_resume.py","test_features.py","test_gpu_mlp.py","test_metrics.py"]
subprocess.run([python,"-m","pytest","-q",*test_files,"--tb=short"],cwd=experiment,env=env,check=True)
print("Integrity tests passed; starting fixed classifier comparison",flush=True)
'''
    execution='''subprocess.run([python,"-u",str(experiment/"run.py"),"--phase","all","--workers","3"],cwd=experiment,env=env,check=True)
'''
    analysis='''subprocess.run([python,"-u",str(experiment/"analyze.py")],cwd=experiment,env=env,check=True)
print((out/"REPORT.md").read_text())
print("Outputs:",sorted(p.name for p in out.iterdir()))
'''
    nb.cells=[nbf.v4.new_markdown_cell('# PROTEUS: classifier-only ablation of the submitted Lag4 forest\n\nPrivate execution. Same fixed geometry,50 trees,subject bootstrap,8 LOSO folds,one later session and original train-only public test. MLP32 runs on CUDA; five literature-grounded alternatives run on CPU. No test tuning. The source bundle and all inputs are hash-verified.'),
              nbf.v4.new_code_cell(bootstrap),
              nbf.v4.new_markdown_cell('## Verify the implementation before accuracy experiments'),
              nbf.v4.new_code_cell(tests),
              nbf.v4.new_markdown_cell('## Run CV,freeze its selection,then evaluate public test'),
              nbf.v4.new_code_cell(execution),
              nbf.v4.new_markdown_cell('## Independently verify metrics and summarize paired differences'),
              nbf.v4.new_code_cell(analysis)]
    nb.metadata['kernelspec']=dict(display_name='Python3',language='python',name='python3')
    nbf.validate(nb);nbf.write(nb,notebook)
    metadata=json.loads((notebook.parent/'kernel-metadata.json').read_text())
    metadata.update(id='den1shh/proteus-lag4-classifier-ablation',title='PROTEUS Lag4 Classifier Ablation',
                    code_file=notebook.name,language='python',kernel_type='notebook',is_private=True,
                    enable_gpu=True,enable_internet=True,machine_shape='NvidiaTeslaT4',
                    dataset_sources=['den1shh/proteus-lag4-classifier-inputs','den1shh/proteus-lag4-classifier-source'],competition_sources=[],kernel_sources=[])
    metadata.pop('id_no',None)
    (notebook.parent/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    receipt=dict(source_files=len(files),source_zip_bytes=len(payload),notebook_bytes=notebook.stat().st_size,
                 source_zip_sha256=hashlib.sha256(payload).hexdigest(),notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),
                 source_files_sha256=hashes,private=True,gpu='NvidiaTeslaT4',dataset=metadata['dataset_sources'][0])
    (HERE/'notebook-build.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_files_sha256'},indent=2))

if __name__=='__main__':main()
