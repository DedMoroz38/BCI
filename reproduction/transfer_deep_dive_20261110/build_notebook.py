"""Small private notebook containing code and saved probabilities, no weights."""
import base64,io,json,zipfile
from pathlib import Path
import nbformat as nbf
from prepare_sources import main as prepare
HERE=Path(__file__).resolve().parent

def main():
    prepare()
    # Keep only probabilities actually needed by this experiment.
    import numpy as np
    for p in (HERE/'reference').glob('*.npz'):
        with np.load(p) as z:
            keep={k:z[k] for k in ('y','subject','session','context','run','onset','data_ids','champion','old_champion','lag_raw')}
        np.savez_compressed(p,**keep)
    files=[HERE/n for n in ('run.py','common.py','nets.py','fusion.py','preprocess.py','averaging.py','tta_core.py','PROTOCOL.md','ORIGINS.json')]
    files+=list((HERE/'reference').glob('*.npz'))+list((HERE/'literature').glob('*.md'))
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.writestr(str(p.relative_to(HERE)),p.read_bytes())
    payload=base64.b64encode(buf.getvalue()).decode()
    setup='''import os,sys,pathlib,io,base64,zipfile,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/transfer-mpl')
work=pathlib.Path('/tmp/transfer-study');work.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode("PAYLOAD"))) as z:z.extractall(work)
envdir=pathlib.Path('/tmp/transfer-env')
subprocess.run([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(envdir)],check=True)
python=str(envdir/'bin/python')
subprocess.run([sys.executable,'-m','pip','--python',python,'install','--quiet','numpy==2.2.6','scipy==1.15.3','scikit-learn==1.7.2','pandas==2.3.3'],check=True)
out=pathlib.Path('/kaggle/working/transfer-results');out.mkdir(parents=True,exist_ok=True)
print((work/'PROTOCOL.md').read_text(),flush=True)
print('No baseline refits, reproduction gates or preliminary test stage.',flush=True)
'''.replace('PAYLOAD',payload)
    execution='''with (out/'execution.log').open('w') as log:
    process=subprocess.Popen([python,'-u','run.py','--minutes','600'],cwd=work,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:
        print(line,end='',flush=True);log.write(line);log.flush()
    code=process.wait()
print('Process exit',code,'saved results',out)
if code:raise RuntimeError('Runner stopped; completed predictions and per-job failures remain in output')
'''
    nb=nbf.read(HERE/'notebook/transfer.ipynb',as_version=4)
    nb.cells=[nbf.v4.new_markdown_cell('# PROTEUS Transfer Normalization Deep Dive\n\n**Goal: high cell BA and transfer to entirely held-out people.** Literature review completed before model implementation.14 fixed small models,2seeds,11outer folds=308fits. Current63.2294% predictions reused;no control refits or parity gates. Private GPU research,no competition submission.'),
              nbf.v4.new_code_cell(setup),nbf.v4.new_markdown_cell('## Source evidence and adaptations'),
              nbf.v4.new_code_cell("print((work/'literature/REVIEW_RU.md').read_text())"),
              nbf.v4.new_markdown_cell('## Run fixed transfer experiments'),nbf.v4.new_code_cell(execution),
              nbf.v4.new_markdown_cell('## Complete-fold results'),nbf.v4.new_code_cell("print((out/'REPORT.md').read_text() if (out/'REPORT.md').exists() else 'See progress.json, execution.log and failure receipts')\nprint('Learning-curve figures:',len(list(out.rglob('curves.png'))))")]
    nbf.write(nb,HERE/'notebook/transfer.ipynb')
    meta=dict(id='den1shh/proteus-transfer-normalization-deep-dive',title='PROTEUS Transfer Normalization Deep Dive',code_file='transfer.ipynb',language='python',kernel_type='notebook',
              is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',
              dataset_sources=['den1shh/proteus-lag4-classifier-inputs'],competition_sources=[],kernel_sources=[])
    (HERE/'notebook/kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print('BUILT private T4 notebook; source bytes', (HERE/'notebook/transfer.ipynb').stat().st_size,';308 outer fits,600min cap')

if __name__=='__main__':main()
