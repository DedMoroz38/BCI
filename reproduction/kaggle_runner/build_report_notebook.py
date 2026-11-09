"""Prepare CPU-only independent verification of completed Kaggle v4 predictions."""
from pathlib import Path
import base64,hashlib,json,zipfile
import nbformat as nbf

here=Path(__file__).resolve().parent;exp=here.parent/'classifier_ablation'
folder=here/'report-notebook';folder.mkdir(exist_ok=True)
with zipfile.ZipFile(here/'code-input/classifier-source.zip') as z:
    original_manifest=z.read('source-integrity.json')
source_sha=hashlib.sha256(original_manifest).hexdigest()
config_files=['PROTOCOL.md','classifiers.py','gpu_mlp.py','common.py','prepare_features.py','run.py']
config=hashlib.sha256(json.dumps({n:hashlib.sha256((exp/n).read_bytes()).hexdigest() for n in config_files},sort_keys=True).encode()).hexdigest()
analyzer=base64.b64encode((exp/'analyze.py').read_bytes()).decode()
tests=base64.b64encode((exp/'test_analysis.py').read_bytes()).decode()
bootstrap=f'''import os,sys,pathlib,json,hashlib,shutil,subprocess,base64
os.environ.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",MPLCONFIGDIR="/tmp/classifier-report-mpl")
selections=list(pathlib.Path("/kaggle/input").rglob("selection.json"))
assert len(selections)==1,selections
previous=selections[0].parent;root=previous.parent
source=root/"riemann-project"
manifest=(source/"source-integrity.json").read_bytes()
assert hashlib.sha256(manifest).hexdigest()=="{source_sha}"
for rel,digest in json.loads(manifest).items():
    assert hashlib.sha256((source/rel).read_bytes()).hexdigest()==digest,rel
project=pathlib.Path("/kaggle/working/verified-project")
shutil.copytree(source,project,ignore=shutil.ignore_patterns("data","cache","__pycache__"))
experiment=project/"reproduction/classifier_ablation"
(experiment/"analyze.py").write_bytes(base64.b64decode("{analyzer}"))
(experiment/"test_analysis.py").write_bytes(base64.b64decode("{tests}"))
out=pathlib.Path("/kaggle/working/classifier-results")
shutil.copytree(previous,out)
envdir=pathlib.Path("/tmp/classifier-report-env")
subprocess.run([sys.executable,"-m","venv","--without-pip","--system-site-packages",str(envdir)],check=True)
python=str(envdir/"bin/python")
pins=["numpy==2.5.4","scipy==1.18.1","scikit-learn==1.9.1","pyriemann==0.12","pandas==3.1.0","threadpoolctl==3.7.0"]
subprocess.run([sys.executable,"-m","pip","--python",python,"install","--disable-pip-version-check","--quiet",*pins],check=True)
env=dict(os.environ,CLASSIFIER_OUTPUT=str(out))
subprocess.run([python,"-c","import analyze;assert analyze.config_hash()=='{config}'"],cwd=experiment,env=env,check=True)
subprocess.run([python,"-m","pytest","-q","test_analysis.py","--tb=short"],cwd=experiment,env=env,check=True)
(out/"analysis-provenance.json").write_text(json.dumps(dict(training_source="den1shh/proteus-lag4-classifier-ablation",training_version=4,classifier_config_sha256="{config}",original_source_manifest_sha256="{source_sha}",analyzer_sha256=hashlib.sha256((experiment/"analyze.py").read_bytes()).hexdigest(),refitted_models=0),indent=2))
print("Original source hashes verified; unchanged predictions copied; analysis tests passed",flush=True)
'''
analysis='''subprocess.run([python,"-u",str(experiment/"analyze.py")],cwd=experiment,env=env,check=True)
verification=json.loads((out/"verification.json").read_text())
assert verification["verified_jobs"]==70 and verification["exact_baseline_prediction_jobs"]==10
assert verification["independently_recomputed_log_loss_and_brier"]
print((out/"REPORT.md").read_text())
print("All70 saved evaluations independently verified. No model refits.")
'''
nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# PROTEUS classifier comparison: verified report\n\nCPU-only verification of the70 saved evaluations from training notebook version4. All training-source hashes,baseline labels,official classification metrics,probability metrics and frozen CV selection are checked. This notebook also reports descriptive selection stability. It does not refit models. The original training notebook has a final-report import conflict corrected here.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_code_cell(analysis)])
nb.metadata['kernelspec']=dict(display_name='Python3',language='python',name='python3')
nbf.validate(nb);nbf.write(nb,folder/'verified_report.ipynb')
metadata=dict(id='den1shh/proteus-lag4-classifier-report',title='PROTEUS Lag4 Classifier Report',code_file='verified_report.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=True,machine_shape='',dataset_sources=[],competition_sources=[],kernel_sources=['den1shh/proteus-lag4-classifier-ablation'])
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(dict(notebook_bytes=(folder/'verified_report.ipynb').stat().st_size,classifier_config_sha256=config,source_manifest_sha256=source_sha,gpu=False),indent=2))
