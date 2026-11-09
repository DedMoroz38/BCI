"""Private T4 notebook; original data/source are reused without any forest run."""
from pathlib import Path
import base64,hashlib,io,json,zipfile
import nbformat as nbf
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PROJECT=ROOT.parent
files=[*HERE.glob('*.py'),HERE/'PROTOCOL.md',ROOT/'attention_spd_ablation/attention.py']
controls=ROOT/'temporal_spd_research/results/lag4_oas'
files+=list(controls.glob('logreg_*.json'))+list(controls.glob('logreg_*_predictions.npz'))
buffer=io.BytesIO();manifest={}
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:
        rel=str(p.relative_to(PROJECT));data=p.read_bytes();z.writestr(rel,data)
        manifest[rel]=hashlib.sha256(data).hexdigest()
    z.writestr('single-manifest.json',json.dumps(manifest,indent=2))
payload=buffer.getvalue();encoded=base64.b64encode(payload).decode();digest=hashlib.sha256(payload).hexdigest()
bootstrap=f'''import os,sys,pathlib,json,hashlib,zipfile,io,base64,subprocess
os.environ.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONUNBUFFERED="1",MPLCONFIGDIR="/tmp/single-mpl")
import torch
assert torch.cuda.is_available(),"Enable requested Kaggle GPU"
print("GPU",torch.cuda.get_device_name(0),flush=True)
project=pathlib.Path("/kaggle/working/riemann-project");project.mkdir(exist_ok=True)
sources=list(pathlib.Path("/kaggle/input").rglob("source-integrity.json"));assert len(sources)==1
raw=sources[0].read_bytes()
assert hashlib.sha256(raw).hexdigest()=="3e8cf0b952a1a18ab496a89d6a31275522e8efa003ea6b8ed4b9e6467d2a3c81"
for rel,digest in json.loads(raw).items():
    p=pathlib.PurePosixPath(rel);assert not p.is_absolute() and ".." not in p.parts
    data=(sources[0].parent/rel).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
    dest=project/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
payload=base64.b64decode("{encoded}")
assert hashlib.sha256(payload).hexdigest()=="{digest}"
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    manifest=json.loads(z.read("single-manifest.json"))
    for rel,digest in manifest.items():
        p=pathlib.PurePosixPath(rel);assert not p.is_absolute() and ".." not in p.parts
        data=z.read(rel);assert hashlib.sha256(data).hexdigest()==digest
        dest=project/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
repro=project/"reproduction";experiment=repro/"single_attention_ablation"
inputs=list(pathlib.Path("/kaggle/input").rglob("input-integrity.json"));assert len(inputs)==1
input_manifest=json.loads(inputs[0].read_text());input_dir=inputs[0].parent
for name,digest in input_manifest["files_sha256"].items():
    h=hashlib.sha256()
    with (input_dir/name).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    assert h.hexdigest()==digest,name
(repro/"data").mkdir(exist_ok=True)
for split in ["train","val","test"]:
    (repro/"data"/f"proteus_{{split}}.npz").symlink_to(input_dir/f"proteus_{{split}}.npz")
(repro/"temporal_spd_research/cache").symlink_to(input_dir,target_is_directory=True)
envdir=pathlib.Path("/tmp/single-env")
subprocess.run([sys.executable,"-m","venv","--without-pip","--system-site-packages",str(envdir)],check=True)
python=str(envdir/"bin/python")
pins=["numpy==2.5.4","scipy==1.18.1","scikit-learn==1.9.1","pyriemann==0.12","pandas==3.1.0","threadpoolctl==3.7.0"]
subprocess.run([sys.executable,"-m","pip","--python",python,"install","--quiet",*pins],check=True)
out=pathlib.Path("/kaggle/working/single-results");out.mkdir(exist_ok=True)
env=dict(os.environ,SINGLE_OUTPUT=str(out))
(out/"bundle-provenance.json").write_text(json.dumps(dict(single_bundle_sha256="{hashlib.sha256(payload).hexdigest()}",original_source_manifest_sha256=hashlib.sha256(raw).hexdigest(),input_manifest=input_manifest,source_manifest=manifest),indent=2))
print("Verified sources and data. SINGLE models,NO forest,full end-to-end gradients.",flush=True)
'''
cells=[nbf.v4.new_markdown_cell('# PROTEUS: single Lag4 vs end-to-end attention SPD\n\nNo forest. Same41channels,band4–40Hz,lag4 and10folds. Baseline: single OAS+causal tangent+LR. Alternatives: attention Gram+ridge and bounded score exponential. IndependentQ/K. Full gradient through prior,cumulative session reference,tangent/scaler and retained final LR. No discarded auxiliary head. Private T4 execution;no competition submission.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_code_cell('subprocess.run([python,"-m","pytest","-q","--tb=short"],cwd=experiment,env=env,check=True)'),nbf.v4.new_code_cell('subprocess.run([python,"-u",str(experiment/"run_single.py")],cwd=experiment,env=env,check=True)'),nbf.v4.new_code_cell('subprocess.run([python,"-u",str(experiment/"analyze_single.py")],cwd=experiment,env=env,check=True)\nprint((out/"REPORT.md").read_text())')]
nb=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')))
nbf.validate(nb);folder=HERE/'notebook';nbf.write(nb,folder/'single_e2e.ipynb')
meta=dict(id='den1shh/proteus-single-attention-e2e',title='PROTEUS Single Attention End-to-End',code_file='single_e2e.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=['den1shh/proteus-lag4-classifier-inputs','den1shh/proteus-lag4-classifier-source'],competition_sources=[],kernel_sources=[])
(folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
(HERE/'notebook-build.json').write_text(json.dumps(dict(bundle_sha256=hashlib.sha256(payload).hexdigest(),source_manifest=manifest,notebook_sha256=hashlib.sha256((folder/'single_e2e.ipynb').read_bytes()).hexdigest()),indent=2))
print(json.dumps(dict(source_files=len(files),zip_bytes=len(payload),private=True,gpu='T4',models=3,forests=0),indent=2))
