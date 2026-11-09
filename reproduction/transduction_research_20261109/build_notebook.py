"""Embed runner and small saved OOF priors; attach public EEG inputs only."""
import base64
import io
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from scipy.special import softmax
import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
META = ('y', 'subject', 'session', 'context', 'run', 'onset')

def keys(a):
    return list(zip(a['subject'].tolist(), a['session'].tolist(), a['context'].tolist(),
                    a['run'].tolist(), a['onset'].tolist()))

def main():
    reference = HERE/'reference'; reference.mkdir(exist_ok=True)
    allmeta = {k: [] for k in META}
    for split in ('train', 'val', 'test'):
        with np.load(ROOT/'data'/f'proteus_{split}.npz') as z:
            for k in META: allmeta[k].extend(z[k].tolist())
    allmeta = {k: np.asarray(v) for k, v in allmeta.items()}
    index = {key: i for i, key in enumerate(keys(allmeta))}
    wave = ROOT/'metric_research_20261108/experiments/all10_baseline/results'
    swad = ROOT/'regularization_research_20261108/downloads/gpu-v2/regularization-results/models'
    receipt = []
    for folder in sorted(wave.iterdir()):
        if not folder.is_dir(): continue
        with np.load(folder/'lag4.npz') as z: base = {k: z[k] for k in z.files}
        target_keys = keys(base); ids = np.array([index[k] for k in target_keys])
        cnn = []
        for seed in (0, 1):
            parent = swad/folder.name/'swad0.1'/f'seed{seed}'
            with np.load(parent/'prediction.npz') as z:
                cnn_index = {key: i for i, key in enumerate(keys(z))}
            order = [cnn_index[k] for k in target_keys]
            cnn.append(softmax(np.load(parent/'logits.npy')[order]/4, axis=1))
        raw = base['proba']; blend = .7*raw+.3*np.mean(cnn, axis=0)
        np.savez_compressed(reference/f'{folder.name}.npz', **{k: base[k] for k in META},
                            data_ids=ids, priors=np.stack([raw, blend]))
        receipt.append(dict(tag=folder.name, windows=len(ids), prior_sources=[str(folder/'lag4.npz'), str(swad/folder.name/'swad0.1')]))
    (HERE/'input_sources.json').write_text(json.dumps(receipt, indent=2))
    vendor = HERE/'vendor'; vendor.mkdir(exist_ok=True)
    author = ROOT/'ensemble_author_20261109'
    shutil.copyfile(author/'track2_bci/transduction/method.py', vendor/'author_method.py')
    shutil.copyfile(author/'LICENSE', vendor/'LICENSE')
    files = [HERE/name for name in ('configs.py', 'core.py', 'run.py', 'PROTOCOL.md', 'input_sources.json')]
    files += sorted(reference.glob('*.npz'))+sorted(vendor.iterdir())
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in files: z.writestr(str(path.relative_to(HERE)), path.read_bytes())
    encoded = base64.b64encode(payload.getvalue()).decode()
    bootstrap = '''import base64, io, os, pathlib, zipfile
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
work=pathlib.Path('/tmp/proteus-transduction-study');work.mkdir(parents=True,exist_ok=True)
payload=base64.b64decode("PAYLOAD")
with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(work)
out=pathlib.Path('/kaggle/working/transduction-results');out.mkdir(parents=True,exist_ok=True)
print('Ready: saved OOF priors, 6 research directions, CPU, 240-minute cap; no classifier training.')
print((work/'PROTOCOL.md').read_text())
'''.replace('PAYLOAD', encoded)
    execution = '''import subprocess, sys
with (out/'execution.log').open('w') as log:
    process=subprocess.Popen([sys.executable,'-u','run.py','--workers','4','--minutes','240'],cwd=work,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:
        print(line,end='',flush=True);log.write(line);log.flush()
    code=process.wait()
if code:raise RuntimeError(f'Study exited with code {code}; completed fold outputs remain in {out}')
'''
    nb = nbf.read(HERE/'notebook/transduction.ipynb', as_version=4)
    nb.cells = [nbf.v4.new_markdown_cell('# PROTEUS causal transduction: six research directions\n\n118 fixed transduction settings, four stability/fusion variants and the saved fallback, each on Lag4 and Lag4+SWAD. 11 outer folds, 2706 evaluations; **no classifier retraining**. CPU execution preserves GPU quota. Target labels are used only for scoring. Public-CV results are exploratory. No reproduction gates or competition submission.'),
                nbf.v4.new_code_cell(bootstrap),
                nbf.v4.new_markdown_cell('## Run the full fixed grid\n\nCompleted folds are saved independently. No separate controls, pytest stage, numerical parity threshold or preliminary model training.'),
                nbf.v4.new_code_cell(execution),
                nbf.v4.new_markdown_cell('## Results\n\nOOF cell BA ranks complete ten-person results. Later-session scores and person deltas are reported separately. There are no epoch loss curves because no classifier is trained.'),
                nbf.v4.new_code_cell("from IPython.display import display, Image\nreport=out/'REPORT.md'\nprint(report.read_text() if report.exists() else (out/'completion.json').read_text() if (out/'completion.json').exists() else 'No complete folds; see execution.log and failures.json')\nfor name in ('top20.png','best_person_deltas.png'):\n    if (out/name).exists():display(Image(filename=str(out/name)))")]
    nbf.write(nb, HERE/'notebook/transduction.ipynb')
    meta = dict(id='den1shh/proteus-causal-transduction-research', title='PROTEUS Causal Transduction Research',
                code_file='transduction.ipynb', language='python', kernel_type='notebook', is_private=True,
                enable_gpu=False, enable_tpu=False, enable_internet=False,
                dataset_sources=['den1shh/proteus-lag4-classifier-inputs'], competition_sources=[], kernel_sources=[])
    (HERE/'notebook/kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print('BUILT', len(receipt), 'folds;', len(payload.getvalue()), 'payload bytes; private CPU notebook')

if __name__ == '__main__': main()
