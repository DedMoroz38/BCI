"""Package explicit reviewed source allowlist; private notebook, no publishing."""
from pathlib import Path
import base64,io,json,zipfile
import numpy as np
import nbformat as nbf

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'custom_validation_20261110/downloads/v1/custom-validation-results'
FILES=('base.py','core.py','neural_training.py','transfer_nets.py','averaging.py',
       'evaluate.py','novel.py','gpu.py','test_novel.py','splits.json','PROTOCOL.md','SWAD_LICENSE')

def main():
    baselines=HERE/'baseline_inputs';baselines.mkdir(exist_ok=True)
    folds=json.loads((HERE/'splits.json').read_text())['folds']
    for tag,record in folds.items():
        source=PRIOR/'predictions'/tag/'full_6458_fixed30_recipe.npz'
        with np.load(source) as z:arrays={k:z[k] for k in z.files}
        arrays['data_ids']=np.asarray(record['test_ids'])
        np.savez_compressed(baselines/f'{tag}.npz',**arrays)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name in FILES:z.writestr(name,(HERE/name).read_bytes())
        for f in sorted(baselines.glob('*.npz')):z.writestr(str(f.relative_to(HERE)),f.read_bytes())
    payload=base64.b64encode(buf.getvalue()).decode()
    setup='''import os,sys,pathlib,io,zipfile,base64,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/literature-mpl',XDG_CACHE_HOME='/tmp/literature-cache')
work=pathlib.Path('/tmp/proteus-literature-wave');work.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode('PAYLOAD'))) as z:z.extractall(work)
envdir=pathlib.Path('/tmp/literature-env')
subprocess.run([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(envdir)],check=True)
python=str(envdir/'bin/python')
subprocess.run([sys.executable,'-m','pip','--python',python,'install','--quiet','numpy==2.2.6','scipy==1.15.3','scikit-learn==1.7.2','pandas==2.3.3','pytest'],check=True)
out=pathlib.Path('/kaggle/working/literature-wave-results');out.mkdir(parents=True,exist_ok=True)
print((work/'PROTOCOL.md').read_text(),flush=True)
subprocess.run([python,'-m','pytest','-q','test_novel.py','--tb=short'],cwd=work,check=True)
'''.replace('PAYLOAD',payload)
    execute='''import signal,json,time
started=time.monotonic();expired=False
with (out/'execution.log').open('w') as log:
    process=subprocess.Popen([python,'-u','gpu.py','--minutes','110'],cwd=work,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('Training started; progress and curves saved in output directory.',flush=True)
    try:
        code=process.wait(timeout=115*60)
    except subprocess.TimeoutExpired:
        expired=True
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
        code=process.returncode
(out/'watchdog.json').write_text(json.dumps(dict(expired=expired,returncode=code,seconds=time.monotonic()-started)))
if expired:
    # No refits: summarize only already completed checkpoints/predictions.
    subprocess.run([python,'-u','gpu.py','--combine-only'],cwd=work,timeout=290,check=True)
elif code:raise RuntimeError(f'Experiment exit code {code}; completed artifacts retained')
'''
    path=HERE/'notebook/literature_wave.ipynb'
    nb=nbf.read(path,as_version=4)
    nb.cells=[nbf.v4.new_markdown_cell('# PROTEUS: Literature-Driven Causal Normalization\n\nTwo new mechanisms,4fixed policies×2seeds×10fixed8/2folds. Source-only inner selection/refit. First-session equal-person scoring. Compare standalone and fixed10/20% additions to already trained8person full ensemble. Private T4;120min application cap. No competition submission or old baseline refit.'),
              nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(execute),
              nbf.v4.new_code_cell("print((out/'validation/REPORT.md').read_text() if (out/'validation/REPORT.md').exists() else 'See progress.json and failures')")]
    nbf.validate(nb);nbf.write(nb,path)
    meta=dict(id='den1shh/proteus-literature-causal-wave',title='PROTEUS Literature Causal Wave',
              code_file=path.name,language='python',kernel_type='notebook',is_private=True,
              enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',
              dataset_sources=['den1shh/proteus-lag4-classifier-inputs'],competition_sources=[],kernel_sources=[])
    (path.parent/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(dict(file=str(path),bytes=path.stat().st_size,outer_fits=80,minutes=120,private=True)))

if __name__=='__main__':main()
