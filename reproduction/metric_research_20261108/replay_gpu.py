"""Replay every saved GPU model on CPU with freshly reconstructed alignment."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
from pathlib import Path
import sys
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'src'))
from common import ROOT, write_json, softmax
from neural_training import align_with_prior, logits
from bci.data import concat, load_split
from bci.stream import ShallowNet


def main():
    start = time.monotonic(); torch.set_num_threads(2)
    root = HERE/'downloads/gpu-v2/metric-results/models'
    full = concat(*(load_split(split, ROOT/'data') for split in ('train','val','test')))
    keys = list(zip(full.subject, full.session, full.run, full.onset))
    lookup = {key:i for i,key in enumerate(keys)}; assert len(lookup) == len(full)
    rows = []
    with threadpool_limits(limits=1, user_api='blas'):
        for fold in sorted(root.iterdir()):
            if not fold.is_dir(): continue
            fits = sorted(fold.glob('*/seed*/prediction.npz')); assert len(fits) == 5
            with np.load(fits[0]) as f:
                idx = np.array([lookup[k] for k in zip(f['subject'],f['session'],f['run'],f['onset'])])
            test = full.select(idx)
            with np.load(fits[0].parent/'alignment.npz') as a: prior = a['cov_prior'].copy()
            x = torch.as_tensor(align_with_prior(test, prior))
            for path in fits:
                with np.load(path.parent/'alignment.npz') as a:
                    np.testing.assert_array_equal(a['cov_prior'],prior)
                    assert int(a['prior_weight']) == 5
                    np.testing.assert_array_equal(a['band'],[1,58])
                saved = torch.load(path.parent/'model.pt',map_location='cpu',weights_only=True)
                net = ShallowNet(**saved['config']); net.load_state_dict(saved['state']); net.eval()
                p = softmax(logits(net, x),axis=1)
                with np.load(path) as f:
                    for name in ('y','subject','session','context','run','onset'):
                        np.testing.assert_array_equal(getattr(test,name),f[name])
                    np.testing.assert_allclose(p,f['proba'],rtol=0,atol=1e-4)
                    np.testing.assert_array_equal(p.argmax(1),f['pred'])
                    delta = float(np.max(np.abs(p-f['proba'])))
                rows.append(dict(path=str(path.relative_to(root)),max_probability_delta=delta,labels_exact=True))
            print('CPU_REPLAY',fold.name,'models',len(fits),flush=True)
    assert len(rows) == 55
    write_json(HERE/'to_human/GPU_CPU_REPLAY.json',dict(models=55,labels_exact=True,
        probability_tolerance=1e-4,max_probability_delta=max(r['max_probability_delta'] for r in rows),
        seconds=time.monotonic()-start,rows=rows))
    print('ALL55 CPU REPLAYS PASSED',round(time.monotonic()-start,2),flush=True)


if __name__ == '__main__': main()
