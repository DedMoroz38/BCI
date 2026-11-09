"""Build a private reproducible notebook with live-updating loss plots."""
from pathlib import Path
import base64,hashlib,io,json,re,zipfile
import nbformat as nbf
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PROJECT=ROOT.parent
files=[*HERE.glob('*.py'),HERE/'PROTOCOL.md']
files += [p for p in (HERE/'vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [*list((ROOT/'single_attention_ablation').glob('*.py')),ROOT/'single_attention_ablation/PROTOCOL.md',ROOT/'attention_spd_ablation/attention.py']
control=ROOT/'temporal_spd_research/results/lag4_oas'
files += list(control.glob('logreg_*.json'))+list(control.glob('logreg_*_predictions.npz'))
buffer=io.BytesIO();manifest={}
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(files)):
        rel=str(p.relative_to(PROJECT));data=p.read_bytes();z.writestr(rel,data);manifest[rel]=hashlib.sha256(data).hexdigest()
    z.writestr('it-manifest.json',json.dumps(manifest,indent=2))
payload=buffer.getvalue();digest=hashlib.sha256(payload).hexdigest()
old=nbf.read(ROOT/'single_attention_ablation/notebook/single_e2e.ipynb',as_version=4)
bootstrap=old.cells[1].source
bootstrap=re.sub(r'payload=base64.b64decode\("[^"\n]+"\)',lambda m:'payload=base64.b64decode("'+base64.b64encode(payload).decode()+'")',bootstrap)
bootstrap=re.sub(r'assert hashlib.sha256\(payload\).hexdigest\(\)=="[^"]+"','assert hashlib.sha256(payload).hexdigest()=="'+digest+'"',bootstrap)
bootstrap=bootstrap.replace('single-manifest.json','it-manifest.json').replace('experiment=repro/"single_attention_ablation"','experiment=repro/"itransformer_ablation"')
bootstrap=bootstrap.replace('/tmp/single-env','/tmp/it-env').replace('/tmp/single-mpl','/tmp/it-mpl').replace('/kaggle/working/single-results','/kaggle/working/it-results')
bootstrap=bootstrap.replace('SINGLE_OUTPUT=str(out)','IT_OUTPUT=str(out)')
bootstrap=re.sub(r'single_bundle_sha256="[^"]+"','it_bundle_sha256="'+digest+'"',bootstrap)
bootstrap=bootstrap.replace('"threadpoolctl==3.7.0"]','"threadpoolctl==3.7.0","einops==0.8.2"]')
bootstrap=bootstrap.replace('Verified sources and data. SINGLE models,NO forest,full end-to-end gradients.','Verified pinned author iTransformer and inputs.41channels,480samples,NO lag copies;single Lag4 baseline.')
execution='''from IPython.display import display, Image
figures={}
process=subprocess.Popen([python,"-u",str(experiment/"run_itransformer.py")],cwd=experiment,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
for line in process.stdout:
    print(line,end="",flush=True)
    if line.startswith("CURVE "):
        figure=pathlib.Path(line.strip()[6:])
        if figure.exists():
            picture=Image(filename=str(figure))
            if str(figure) in figures:figures[str(figure)].update(picture)
            else:figures[str(figure)]=display(picture,display_id=True)
code=process.wait()
if code:raise RuntimeError(f"Experiment exited with status {code};partial curves and checkpoints are saved")
'''
nb=nbf.read(HERE/'notebook/itransformer_lag4.ipynb',as_version=4)
nb.cells=[nbf.v4.new_markdown_cell('# PROTEUS: author iTransformer without lags vs single Lag4\n\nPrivate GPU experiment;no competition submission. Article-aligned8LOSO,later-session and old public test.\n\n**iTransformer:** official pinned encoder c2426e68;41tokens of480samples;one block,d64,4heads,FFN128,GELU,dropout0.2;flatten+linear3-class head. AdamW0.001,decay0.001,batch64,max80epochs,patience10. Seeds0,1,2 reported separately.\n\n**Baseline:** exact single Lag4 OAS+causal reference+LR;41electrodes expanded to82rows,prior5,C0.01.\n\nInner validation people never enter fitting or normalization. Best epoch selected on inner validation weighted CE;fresh refit on all outer-training people. Curves below update during inner selection;outer evaluation never appears as validation.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_markdown_cell('## Verify author equivalence,split isolation,checkpoint selection and integrity'),nbf.v4.new_code_cell('subprocess.run([python,"-m","pytest","-q","--tb=short"],cwd=experiment,env=env,check=True)'),nbf.v4.new_markdown_cell('## Fit models and show train/validation curves\n\nEach figure is an inner person-held-out fit. Blue=train,orange=validation,both in eval mode. Dotted lines show the inner Lag4 baseline;vertical marker shows selected epoch. CSV saved each epoch;PNG updated every5epochs and at completion. Fresh outer refit has its own training-history CSV.'),nbf.v4.new_code_cell(execution),nbf.v4.new_markdown_cell('## Independent metric verification and final comparison'),nbf.v4.new_code_cell('subprocess.run([python,"-u",str(experiment/"analyze_itransformer.py")],cwd=experiment,env=env,check=True)\nprint((out/"REPORT.md").read_text())')]
nbf.validate(nb);nbf.write(nb,HERE/'notebook/itransformer_lag4.ipynb')
meta=dict(id='den1shh/proteus-itransformer-vs-lag4',title='PROTEUS iTransformer vs Lag4',code_file='itransformer_lag4.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=['den1shh/proteus-lag4-classifier-inputs','den1shh/proteus-lag4-classifier-source'],competition_sources=[],kernel_sources=[])
(HERE/'notebook/kernel-metadata.json').write_text(json.dumps(meta,indent=2))
(HERE/'notebook-build.json').write_text(json.dumps(dict(bundle_sha256=digest,source_manifest=manifest,notebook_sha256=hashlib.sha256((HERE/'notebook/itransformer_lag4.ipynb').read_bytes()).hexdigest()),indent=2))
print(json.dumps(dict(source_files=len(files),zip_bytes=len(payload),private=True,gpu='T4',families=2,neural_seeds=3),indent=2))
