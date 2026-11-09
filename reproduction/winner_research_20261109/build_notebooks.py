"""Package private CPU/GPU notebooks. No reproduction gates in execution."""
import ast,base64,io,json,shutil,zipfile
from pathlib import Path
import numpy as np
import nbformat as nbf
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def prepare():
    shutil.copyfile(ROOT/'transduction_research_20261109/core.py',HERE/'tta_core.py')
    source=(ROOT/'regularization_research_20261108/src/regularizers.py').read_text();tree=ast.parse(source)
    names={'clone_state','mean_states','StateAverage','LossValley'}
    code='"""Averaging policies from our prior SWAD experiment, unchanged."""\nfrom collections import deque\nimport torch\nimport numpy as np\n\n'
    code+='\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)
    (HERE/'averaging.py').write_text(code+'\n')
    (HERE/'reference').mkdir(exist_ok=True)
    previous=ROOT/'transduction_research_20261109'
    for tag in ['loso_P'+f'{v:02}' for v in range(1,11)]+['later_last']:
        with np.load(previous/'reference'/f'{tag}.npz') as z:ref={k:z[k] for k in z.files}
        with np.load(previous/'downloads/v1/transduction-results/folds'/f'{tag}.npz') as z:
            names=z['names'].tolist();pi=z['priors'].tolist().index('lag4_swad');p=z['proba']
            arrays=dict(champion=p[names.index('lag_running__soft50_half'),pi],stable=p[names.index('views_stable005'),pi],
                        fallback=p[names.index('fallback_balance1'),pi],full_trans=p[names.index('lag_running__soft50'),pi])
        np.savez_compressed(HERE/'reference'/f'{tag}.npz',**{k:v for k,v in ref.items() if k!='priors'},
                            lag_raw=ref['priors'][0],mixed_raw=ref['priors'][1],**arrays)
        for seed in (0,1):
            old=ROOT/'regularization_research_20261108/downloads/gpu-v2/regularization-results/models'/tag/'swad0.1'/f'seed{seed}'
            dest=HERE/'frozen'/tag/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
            for name in ('model.pt','alignment.npz'):shutil.copyfile(old/name,dest/name)
    shutil.copyfile(ROOT/'EXPERIMENT_PLAN_2026_11_09.md',HERE/'EXPERIMENT_PLAN.md')
    provenance={}
    for name in ('TSMNet','minirocket'):
        import subprocess
        provenance[name]=dict(url='https://github.com/'+('rkobler/TSMNet' if name=='TSMNet' else 'angus924/minirocket'),
                             commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE/'vendor'/name,text=True).strip())
    (HERE/'SOURCE_ORIGINS.json').write_text(json.dumps(provenance,indent=2))
    for old in (ROOT/'regularization_research_20261108/literature/vendor').glob('*LICENSE*'):
        shutil.copyfile(old,HERE/old.name)

def bundle(mode):
    files=[HERE/n for n in ('common.py','cpu.py','gpu.py','nets.py','tta_core.py','averaging.py','EXPERIMENT_PLAN.md','EXECUTION.md','SOURCE_ORIGINS.json')]
    files+=list((HERE/'reference').glob('*.npz'))
    # Saved CNN weights are attached from their existing Kaggle output;
    # embedding them would exceed Kaggle's 1 MB source limit.
    for name in ('__init__.py','modules.py','functionals.py','manifolds.py'):
        files.append(HERE/'vendor/TSMNet/spdnets'/name)
    files += [HERE/'vendor/TSMNet/LICENSE',HERE/'vendor/TSMNet/experiments/conf/nnet/tsmnet.yaml',
              HERE/'vendor/minirocket/LICENSE',HERE/'vendor/minirocket/code/minirocket_multivariate.py']
    files += list(HERE.glob('*LICENSE*'))
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.writestr(str(path.relative_to(HERE)),path.read_bytes())
    return base64.b64encode(buf.getvalue()).decode()

def main():
    prepare()
    pins=['numpy==2.2.6','scipy==1.15.3','scikit-learn==1.7.2','pandas==2.3.3','pyriemann==0.9','numba==0.61.2','geoopt==0.5.1']
    for mode,title,slug,minutes,timeout in [('cpu','PROTEUS Winner Features CPU','proteus-winner-features-cpu',240,15000),
                                           ('gpu','PROTEUS Causal Alignment GPU','proteus-causal-alignment-gpu',690,43200)]:
        folder=HERE/f'notebook-{mode}';name=f'winner_{mode}.ipynb';payload=bundle(mode)
        bootstrap='''import os,sys,pathlib,io,base64,zipfile,subprocess,json
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',MPLCONFIGDIR='/tmp/winner-mpl')
work=pathlib.Path('/tmp/winner-study');work.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode("PAYLOAD"))) as z:z.extractall(work)
envdir=pathlib.Path('/tmp/winner-env')
subprocess.run([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(envdir)],check=True)
python=str(envdir/'bin/python')
subprocess.run([sys.executable,'-m','pip','--python',python,'install','--quiet',*PINS],check=True)
out=pathlib.Path('/kaggle/working/winner-MODE-results');out.mkdir(parents=True,exist_ok=True)
print((work/'EXECUTION.md').read_text(),flush=True)
print('Actual research starts next. No control fits, no pytest stage, no historical tolerance gate.',flush=True)
'''.replace('PAYLOAD',payload).replace('PINS',repr(pins)).replace('MODE',mode)
        if mode=='cpu':
            bootstrap+='''frozen=work/'frozen'
count=0
for path in pathlib.Path('/kaggle/input').rglob('model.pt'):
    if path.parent.parent.name!='swad0.1':continue
    tag=path.parent.parent.parent.name;seed=path.parent.name
    dest=frozen/tag/seed;dest.mkdir(parents=True,exist_ok=True)
    for name in ('model.pt','alignment.npz'):
        link=dest/name
        if not link.exists():link.symlink_to(path.parent/name)
    count+=1
print('Attached saved SWAD CNN models:',count,flush=True)
'''
        execution=f'''with (out/'execution.log').open('w') as log:
    process=subprocess.Popen([python,'-u','{mode}.py','--minutes','{minutes}'],cwd=work,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:
        print(line,end='',flush=True);log.write(line);log.flush()
    code=process.wait()
print('Process exit:',code,'; saved results:',out)
if code:raise RuntimeError('Runner stopped; completed predictions and failures are preserved in output')
'''
        nb=nbf.read(folder/name,as_version=4)
        nb.cells=[nbf.v4.new_markdown_cell('# '+title+'\n\nPrivate research on all 10 people: 10 LOSO + later_last. Historical full baseline 62.7234% OOF is reused. No baseline retraining, parity checks, preliminary test suite or competition submission. Source and predictions are saved; public CV is exploratory.'),
                  nbf.v4.new_code_cell(bootstrap),nbf.v4.new_markdown_cell('## Run the fixed research package'),nbf.v4.new_code_cell(execution),
                  nbf.v4.new_markdown_cell('## Complete-fold comparison'),nbf.v4.new_code_cell("print((out/'REPORT.md').read_text() if (out/'REPORT.md').exists() else 'See execution.log, progress.json and failure receipts')\nprint('Curves:',len(list(out.rglob('curves.png'))))")]
        nbf.write(nb,folder/name)
        metadata=dict(id='den1shh/'+slug,title=title,code_file=name,language='python',kernel_type='notebook',is_private=True,
                      enable_gpu=mode=='gpu',enable_tpu=False,enable_internet=True,dataset_sources=['den1shh/proteus-lag4-classifier-inputs'],competition_sources=[],kernel_sources=[])
        if mode=='gpu':metadata['machine_shape']='NvidiaTeslaT4'
        else:metadata['kernel_sources']=['den1shh/proteus-regularization-wave2']
        (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
        print('BUILT',mode,'bytes', (folder/name).stat().st_size,'private','timeout',timeout,flush=True)

if __name__=='__main__':main()
