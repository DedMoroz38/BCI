"""Private GPU validation; bundle only newly trained8person classical priors."""
from pathlib import Path
import io,base64,json,zipfile,shutil
import numpy as np
import nbformat as nbf

HERE=Path(__file__).resolve().parent

def main():
    splits=json.loads((HERE/'splits.json').read_text())['folds']
    thin=HERE/'classical_inputs';thin.mkdir(exist_ok=True)
    names=['lag4_matched_lr50','lag4_nuclear001_50','lag4_nuclear005_50','mssa8_mix25_lr50']
    for tag in splits:
        arrays={}
        for name in names:
            p=HERE/'results_pair_cpu/predictions'/tag/(name+'.npz')
            with np.load(p) as z:
                arrays[name]=z['proba']
                if 'y' not in arrays:arrays.update({k:z[k] for k in ('y','subject','session','context','run','onset')})
        arrays['data_ids']=np.asarray(splits[tag]['test_ids'])
        np.savez_compressed(thin/f'{tag}.npz',**arrays)
    files=[HERE/n for n in ('base.py','core.py','gpu.py','neural_training.py','transfer_nets.py','averaging.py','tta_core.py','window_cache.py','evaluate.py','splits.json','PROTOCOL.md','SWAD_LICENSE')]
    files+=list(thin.glob('*.npz'))
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.writestr(str(p.relative_to(HERE)),p.read_bytes())
    payload=base64.b64encode(buf.getvalue()).decode()
    setup='''import os,sys,pathlib,io,zipfile,base64,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/custom-val-mpl')
work=pathlib.Path('/tmp/custom-validation');work.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode('PAYLOAD'))) as z:z.extractall(work)
envdir=pathlib.Path('/tmp/custom-val-env')
subprocess.run([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(envdir)],check=True)
python=str(envdir/'bin/python')
subprocess.run([sys.executable,'-m','pip','--python',python,'install','--quiet','numpy==2.2.6','scipy==1.15.3','scikit-learn==1.7.2','pandas==2.3.3'],check=True)
out=pathlib.Path('/kaggle/working/custom-validation-results');out.mkdir(parents=True,exist_ok=True)
print((work/'PROTOCOL.md').read_text(),flush=True)
print('Only new8person classical priors included; no9person weights or old LOSO forecasts.',flush=True)
'''.replace('PAYLOAD',payload)
    execute='''with (out/'execution.log').open('w') as log:
    process=subprocess.Popen([python,'-u','gpu.py','--minutes','180'],cwd=work,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:
        print(line,end='',flush=True);log.write(line);log.flush()
    code=process.wait()
print('Exit code:',code)
if code:raise RuntimeError('See saved logs,completed fits and failure receipts')
'''
    path=HERE/'notebook/custom_validation.ipynb';nb=nbf.read(path,as_version=4)
    nb.cells=[nbf.v4.new_markdown_cell('# PROTEUS Custom Participant Validation\n\nApproved first-session equal-person scoring,10fixed8/2folds,stream bins1–20/21–60/61+. Existing top recipes,60neural outer fits,2seeds. New source-only priors,private GPU,no submission or historical parity gates.'),
              nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(execute),
              nbf.v4.new_code_cell("print((out/'validation/REPORT.md').read_text() if (out/'validation/REPORT.md').exists() else 'See progress.json and failures')")]
    nbf.write(nb,path)
    meta=dict(id='den1shh/proteus-custom-person-validation',title='PROTEUS Custom Person Validation',code_file=path.name,language='python',kernel_type='notebook',is_private=True,
              enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=['den1shh/proteus-lag4-classifier-inputs'],competition_sources=[],kernel_sources=[])
    (path.parent/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print('BUILT privateT4;bytes',path.stat().st_size,'60outerfits;180min application cap;10new source pairs')

if __name__=='__main__':main()
