"""Embed only new code; reuse immutable private original inputs/source datasets."""
from pathlib import Path
import base64,hashlib,io,json,zipfile
import nbformat as nbf
HERE=Path(__file__).resolve().parent
files=[*HERE.glob('*.py'),HERE/'PROTOCOL.md']
buffer=io.BytesIO();manifest={}
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:
        data=p.read_bytes();z.writestr(p.name,data);manifest[p.name]=hashlib.sha256(data).hexdigest()
    z.writestr('attention-manifest.json',json.dumps(manifest,indent=2))
payload=buffer.getvalue();encoded=base64.b64encode(payload).decode()
bootstrap=f'''import os,sys,pathlib,json,hashlib,zipfile,io,base64,subprocess
os.environ.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONUNBUFFERED="1",MPLCONFIGDIR="/tmp/attention-mpl")
import torch
assert torch.cuda.is_available(),"Requested experiment requires Kaggle GPU"
print("GPU",torch.cuda.get_device_name(0),flush=True)
project=pathlib.Path("/kaggle/working/riemann-project");project.mkdir(exist_ok=True)
sources=list(pathlib.Path("/kaggle/input").rglob("source-integrity.json"));assert len(sources)==1,sources
raw=sources[0].read_bytes()
assert hashlib.sha256(raw).hexdigest()=="3e8cf0b952a1a18ab496a89d6a31275522e8efa003ea6b8ed4b9e6467d2a3c81"
for rel,digest in json.loads(raw).items():
    p=pathlib.PurePosixPath(rel);assert not p.is_absolute() and ".." not in p.parts
    data=(sources[0].parent/rel).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
    dest=project/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
repro=project/"reproduction";experiment=repro/"attention_spd_ablation";experiment.mkdir(exist_ok=True)
payload=base64.b64decode("{encoded}")
assert hashlib.sha256(payload).hexdigest()=="{hashlib.sha256(payload).hexdigest()}"
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    manifest=json.loads(z.read("attention-manifest.json"))
    for name,digest in manifest.items():
        assert pathlib.PurePosixPath(name).name==name
        data=z.read(name);assert hashlib.sha256(data).hexdigest()==digest
        (experiment/name).write_bytes(data)
(experiment/"attention-manifest.json").write_text(json.dumps(manifest,indent=2))
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
envdir=pathlib.Path("/tmp/attention-env")
subprocess.run([sys.executable,"-m","venv","--without-pip","--system-site-packages",str(envdir)],check=True)
python=str(envdir/"bin/python")
pins=["numpy==2.5.4","scipy==1.18.1","scikit-learn==1.9.1","pyriemann==0.12","pandas==3.1.0","threadpoolctl==3.7.0"]
subprocess.run([sys.executable,"-m","pip","--python",python,"install","--quiet",*pins],check=True)
out=pathlib.Path("/kaggle/working/attention-results");out.mkdir(exist_ok=True)
env=dict(os.environ,ATTENTION_OUTPUT=str(out),ATTENTION_CACHE="/tmp/attention-cache")
(out/"bundle-provenance.json").write_text(json.dumps(dict(attention_bundle_sha256="{hashlib.sha256(payload).hexdigest()}",original_source_manifest_sha256=hashlib.sha256(raw).hexdigest(),input_manifest=input_manifest,source_manifest=manifest),indent=2))
print("Original and new source plus all inputs verified",flush=True)
'''
# Path.symlink_to uses target_is_directory (not target_is_directory alias on link creation).
cells=[nbf.v4.new_markdown_cell('# PROTEUS: learned attention SPD vs submitted Lag4\n\nPrivate GPU run. Independent W_Q and W_K. Two fixed supervised representations,50 independent trees each,exact old10fold evaluation. Encoder trained only on each tree\'s bootstrap subjects. Original causal reference/scaler/LR refitted after freezing encoder. No competition submission.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_code_cell('subprocess.run([python,"-m","pytest","-q","--tb=short"],cwd=experiment,env=env,check=True)'),nbf.v4.new_code_cell('subprocess.run([python,"-u",str(experiment/"run_attention.py")],cwd=experiment,env=env,check=True)'),nbf.v4.new_code_cell('subprocess.run([python,"-u",str(experiment/"analyze_attention.py")],cwd=experiment,env=env,check=True)\nprint((out/"REPORT.md").read_text())')]
nb=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')))
nbf.validate(nb);folder=HERE/'notebook';folder.mkdir(exist_ok=True)
nbf.write(nb,folder/'attention_spd.ipynb')
meta=dict(id='den1shh/proteus-attention-spd-ablation',title='PROTEUS Attention SPD Ablation',code_file='attention_spd.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=['den1shh/proteus-lag4-classifier-inputs','den1shh/proteus-lag4-classifier-source'],competition_sources=[],kernel_sources=[])
(folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
(HERE/'notebook-build.json').write_text(json.dumps(dict(bundle_sha256=hashlib.sha256(payload).hexdigest(),source_manifest=manifest,notebook_sha256=hashlib.sha256((folder/'attention_spd.ipynb').read_bytes()).hexdigest()),indent=2))
print(json.dumps(dict(source_files=len(files),zip_bytes=len(payload),private=True,gpu=meta['machine_shape']),indent=2))
