"""Run independent variant/model groups in at most four local processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
args = parser.parse_args()
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           MPLCONFIGDIR=str(ROOT / '.cache/matplotlib'))
# Long groups first; every group uses its own process and disjoint result files.
variants = ['kullback_sym', 'wasserstein_recursive', 'poweuclid_half',
            'harmonic_cumulative', 'logchol_cumulative']
jobs = [(v, m) for m in ['forest20', 'forest12', 'logreg'] for v in variants]
(HERE / 'logs').mkdir(exist_ok=True)

def run(job):
    variant, model = job
    log = HERE / 'logs' / f'{variant}_{model}.log'
    print(f'START group {variant}/{model}', flush=True)
    with log.open('w') as stream:
        result = subprocess.run([sys.executable, '-u', str(HERE / 'run.py'),
                                 '--variant', variant, '--model', model],
                                env=env, stdout=stream, stderr=subprocess.STDOUT)
    print(f'{"DONE" if result.returncode == 0 else "FAILED"} group {variant}/{model}, log={log.name}', flush=True)
    return job, result.returncode

failed = []
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    for future in as_completed([pool.submit(run, job) for job in jobs]):
        job, code = future.result()
        if code:
            failed.append((job, code))
if failed:
    raise SystemExit(f'Failed groups: {failed}; inspect logs and resume with the same command')
print('All 145 candidate jobs completed or resumed.', flush=True)
