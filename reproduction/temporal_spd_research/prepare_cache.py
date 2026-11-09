"""Precompute exact label-free matrices once, with resumable per-band jobs."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,json,os
from pathlib import Path
import subprocess,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'scripts'))
from reproduce_eeg import COMMIT,sha256,validate_splits
from temporal_features import CANDIDATES
from temporal_cache import initialize_cache,build_one
from bci.data import load_split,concat
from bci.models.forest import BANDS
from filelock import FileLock


def inputs():
    source=ROOT/'Brain-Decoding'
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=COMMIT or subprocess.check_output(['git','diff','HEAD'],cwd=source):
        raise RuntimeError('Original source must be unchanged')
    baseline=json.loads((ROOT/'results/eeg/manifest.json').read_text())
    hashes={s:sha256(ROOT/'data'/f'proteus_{s}.npz') for s in ['train','val','test']}
    if hashes!=baseline['data_sha256']:raise RuntimeError('Data differ from original control')
    parts={s:load_split(s,ROOT/'data') for s in ['train','val','test']}
    validate_splits(parts)
    return concat(parts['train'],parts['val'],parts['test']).X,hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--representation',choices=CANDIDATES)
    p.add_argument('--band',type=int,choices=range(6))
    p.add_argument('--workers',type=int,choices=range(1,7),default=6)
    p.add_argument('--initialize',action='store_true')
    p.add_argument('--auxiliary',action='store_true',help='Build only delay/envelope while HSIC jobs run')
    args=p.parse_args()
    X,hashes=inputs()
    directory=HERE/'cache'
    initialize_cache(directory,X,BANDS,hashes)
    if args.initialize:return
    if args.representation is not None:
        if args.band is None:p.error('--band is required with --representation')
        with FileLock(str(directory/f'{args.representation}_band{args.band}.lock')):
            build_one(directory,X,args.representation,args.band)
        return
    del X
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
             MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'))
    # Each transform/band is independent and deterministic.
    jobs=[(rep,b) for rep in CANDIDATES for b in range(6)]
    if args.auxiliary:
        jobs=[j for j in jobs if j[0]!='hsic_rbf']
    (HERE/'logs').mkdir(exist_ok=True)
    def run(job):
        rep,b=job
        print(f'START cache {rep} band {b}',flush=True)
        with (HERE/'logs'/f'cache_{rep}_band{b}.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__)),
                '--representation',rep,'--band',str(b)],env=env,stdout=log,stderr=subprocess.STDOUT)
        print(f'{"DONE" if result.returncode==0 else "FAILED"} cache {rep} band {b}',flush=True)
        return job,result.returncode
    failed=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            job,code=future.result()
            if code:failed.append((job,code))
    if failed:raise SystemExit(f'Failed cache jobs: {failed}; inspect logs and resume')
    print('All 36 exact matrix caches complete.',flush=True)

if __name__=='__main__':main()
