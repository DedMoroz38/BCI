"""Private Kaggle CPU study. Saved OOF priors; no classifier fitting or gates."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.signal import butter, sosfiltfilt
from configs import CONFIGS, GEOMETRIES, STABILITY
from core import spectral, gram_for_geometry, causal_geometry, balance, stability_predictions

HERE = Path(__file__).resolve().parent
BANDS = ((4, 8), (8, 13), (13, 20), (20, 30), (4, 40))
META = ('y', 'subject', 'session', 'context', 'run', 'onset')
PRIORS = ('lag4', 'lag4_swad')
NAMES = ['fallback_balance1']+[c['name'] for c in CONFIGS]+list(STABILITY)

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)

def load_data(root):
    parts = []
    for split in ('train', 'val', 'test'):
        paths = sorted(Path(root).rglob('proteus_'+split+'.npz'))
        if not paths: raise FileNotFoundError('Missing public input: proteus_'+split+'.npz')
        with np.load(paths[0]) as z:
            parts.append({k: z[k] for k in ('X',)+META})
    return {k: np.concatenate([p[k] for p in parts]) for k in ('X',)+META}

def prepare_band(args):
    cache, bi = args; cache = Path(cache)
    X = np.load(cache/'X.npy', mmap_mode='r')
    sos = butter(4, BANDS[bi], btype='bandpass', fs=120, output='sos')
    outputs = []
    for lag in (False, True):
        d = X.shape[1]*(2 if lag else 1)
        path = cache/f'logs_{int(lag)}_{bi}.npy'
        outputs.append(np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(len(X), d, d)))
    for first in range(0, len(X), 24):
        x = sosfiltfilt(sos, np.asarray(X[first:first+24], dtype='float64'), axis=-1)
        for lag, dest in enumerate(outputs):
            z = np.stack([x[..., 4:], x[..., :-4]], axis=-2).reshape(len(x), 2*x.shape[1], x.shape[-1]-4) if lag else x
            z = z-z.mean(-1, keepdims=True); n = z.shape[-1]; d = z.shape[1]
            S = z@np.swapaxes(z, -1, -2)/n
            alpha = np.mean(S*S, axis=(-1, -2)); mu = np.trace(S, axis1=-2, axis2=-1)/d
            den = (n+1)*(alpha-mu*mu/d)
            shrink = np.minimum((alpha+mu*mu)/np.maximum(den, 1e-100), 1.)
            S *= (1-shrink[:, None, None]); S += (shrink*mu)[:, None, None]*np.eye(d)
            dest[first:first+len(x)] = spectral(S, 'log')
        if first % 480 == 0: print('CACHE band', BANDS[bi], first, '/', len(X), flush=True)
    for dest in outputs: dest.flush()
    return bi

def stream_groups(meta):
    keys = np.char.add(np.char.add(meta['subject'], '/'), meta['session'])
    for key in np.unique(keys):
        idx = np.flatnonzero(keys == key)
        yield idx[np.lexsort((meta['onset'][idx], meta['run'][idx]))]

def score(meta, p):
    y = meta['y']; pred = p.argmax(-1)
    def ba(idx):
        classes = np.union1d(y[idx], pred[idx])
        return float(np.mean([np.mean(pred[idx][y[idx] == c] == c) if np.any(y[idx] == c) else 0. for c in classes]))
    keys = np.array(['|'.join(map(str, row)) for row in zip(meta['subject'], meta['session'], meta['context'])])
    return dict(cell=float(np.mean([ba(np.flatnonzero(keys == key)) for key in np.unique(keys)])),
                nll=float(-np.mean(np.log(np.maximum(p[np.arange(len(p)), y], 1e-15)))),
                brier=float(np.mean(np.sum((p-np.eye(3)[y])**2, axis=-1))))

def fold_job(args):
    tag, cache, out, deadline = args; cache = Path(cache); out = Path(out)
    start = time.monotonic()
    ref = np.load(HERE/'reference'/f'{tag}.npz')
    meta = {k: ref[k] for k in META}; ids = ref['data_ids']; priors = ref['priors']
    count = len(np.load(cache/'X.npy', mmap_mode='r'))
    train = np.ones(count, bool); train[ids] = False
    source = []; matrices = []
    for rep in (0, 1):
        files = [np.load(cache/f'logs_{rep}_{bi}.npy', mmap_mode='r') for bi in range(len(BANDS))]
        source.append(np.stack([np.mean(a[train], axis=0, dtype='float64') for a in files]))
        matrices.append(np.stack([a[ids] for a in files], axis=1).astype('float64'))
    predictions = np.empty((len(NAMES), 2, len(ids), 3), dtype='float64')
    name_to_id = {v: i for i, v in enumerate(NAMES)}
    for stream_i, idx in enumerate(stream_groups(meta)):
        p = priors[:, idx]; fallback = np.stack([balance(v) for v in p])
        predictions[0][:, idx] = fallback
        logs = [a[idx] for a in matrices]
        for geometry in GEOMETRIES:
            if time.monotonic() > deadline: raise TimeoutError('Study wall budget reached')
            cases = [c for c in CONFIGS if c['geometry'] == geometry]
            grams = gram_for_geometry(logs, source, geometry, p[0])
            q = causal_geometry(grams, p, cases, deadline)
            for i, case in enumerate(cases): predictions[name_to_id[case['name']]][:, idx] = q[i]
            print('DONE', tag, 'session', stream_i+1, geometry, 'configs', len(cases)*2,
                  'seconds', round(time.monotonic()-start, 1), flush=True)
        a = predictions[name_to_id['ordinary_running__soft50']][:, idx]
        b = predictions[name_to_id['lag_running__soft50']][:, idx]
        for name, values in stability_predictions(a, b, fallback).items():
            predictions[name_to_id[name]][:, idx] = values
    path = out/'folds'/f'{tag}.npz'; path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **meta, names=np.array(NAMES), priors=np.array(PRIORS), proba=predictions)
    rows = [dict(tag=tag, config=name, prior=prior, **score(meta, predictions[ci, pi]))
            for ci, name in enumerate(NAMES) for pi, prior in enumerate(PRIORS)]
    write_json(path.with_suffix('.json'), dict(seconds=time.monotonic()-start, scores=rows))
    print('FOLD COMPLETE', tag, round(time.monotonic()-start, 1), 'seconds', flush=True)
    return tag

def summarize(out):
    import pandas as pd
    out = Path(out)
    # The JSON receipt is written last: another worker can still be writing
    # its NPZ or computing scores when this function is called.
    paths = sorted(p for p in (out/'folds').glob('*.npz') if p.with_suffix('.json').exists())
    if not paths: return
    files = {}
    for p in paths:
        with np.load(p) as z:
            files[p.stem] = {k: z[k] for k in z.files}
    foldrows = [r for p in paths for r in json.loads(p.with_suffix('.json').read_text())['scores']]
    pd.DataFrame(foldrows).to_csv(out/'fold_scores.csv', index=False)
    loso = sorted(k for k in files if k.startswith('loso_')); rows = []
    if len(loso) == 10:
        meta = {k: np.concatenate([files[t][k] for t in loso]) for k in META}
        for ci, name in enumerate(NAMES):
            for pi, prior in enumerate(PRIORS):
                p = np.concatenate([files[t]['proba'][ci, pi] for t in loso])
                values = score(meta, p)
                person = [score({k: files[t][k] for k in META}, files[t]['proba'][ci, pi])['cell'] for t in loso]
                late = score({k: files['later_last'][k] for k in META}, files['later_last']['proba'][ci, pi])['cell'] if 'later_last' in files else None
                rows.append(dict(config=name, prior=prior, oof_cell=values['cell'],
                                 mean_person=float(np.mean(person)), later_cell=late, nll=values['nll'], brier=values['brier']))
        df = pd.DataFrame(rows)
        baseline = df[df.config == 'fallback_balance1'].set_index('prior').oof_cell.to_dict()
        df['delta_pp'] = [(v-baseline[p])*100 for v, p in zip(df.oof_cell, df.prior)]
        df = df.sort_values('oof_cell', ascending=False)
        df.to_csv(out/'comparison.csv', index=False)
        lines = ['# Causal transduction research', '',
                 f'{len(files)}/11 outer folds complete; {len(NAMES)} configurations x 2 saved priors.',
                 'Main metric: pooled OOF mean subject/session/context balanced accuracy. Later-session metric is separate.',
                 'Exploratory public-CV comparison; all probabilities emitted from current/past inputs only.', '',
                 df.head(30).to_string(index=False)]
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            top = df.head(20).iloc[::-1]
            fig, ax = plt.subplots(figsize=(11, 8))
            ax.barh(top.config+' / '+top.prior, 100*top.oof_cell)
            ax.set_xlabel('OOF cell balanced accuracy (%)'); ax.set_xlim(45, max(65, 100*top.oof_cell.max()+1))
            fig.tight_layout(); fig.savefig(out/'top20.png', dpi=160); plt.close(fig)
            best = df.iloc[0]; ci = NAMES.index(best.config); pi = PRIORS.index(best.prior)
            deltas = []
            for t in loso:
                m = {k: files[t][k] for k in META}
                deltas.append(100*(score(m, files[t]['proba'][ci, pi])['cell']-score(m, files[t]['proba'][0, pi])['cell']))
            fig, ax = plt.subplots(figsize=(9, 4)); ax.bar(loso, deltas); ax.axhline(0, c='k', lw=.8)
            ax.set_ylabel('Delta vs same-prior fallback (pp)'); ax.tick_params(axis='x', rotation=45)
            fig.tight_layout(); fig.savefig(out/'best_person_deltas.png', dpi=160); plt.close(fig)
        except Exception as error: print('Plot skipped:', error, flush=True)
        print('\n'.join(lines), flush=True)
    write_json(out/'completion.json', dict(completed_folds=sorted(files), planned_folds=11,
                                          configs_per_prior=len(NAMES), priors=PRIORS, full_oof=len(loso) == 10))

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data-root', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--out', type=Path, default=Path('/kaggle/working/transduction-results'))
    parser.add_argument('--cache', type=Path, default=Path('/tmp/transduction-cache'))
    parser.add_argument('--minutes', type=float, default=240); parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)
    start = time.monotonic(); deadline = start+args.minutes*60
    write_json(args.out/'protocol.json', dict(configs=CONFIGS, stability=STABILITY, bands=BANDS, priors=PRIORS,
        wall_cap_minutes=args.minutes, workers=args.workers, classifier_training=False,
        target_labels='Scoring only, never supplied to transduction or geometry'))
    data = load_data(args.data_root); np.save(args.cache/'X.npy', data.pop('X')); del data
    print('PREPARE label-free ordinary/Lag4 log-covariances', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for bi in pool.map(prepare_band, [(str(args.cache), b) for b in range(len(BANDS))]):
            print('CACHE COMPLETE band', BANDS[bi], flush=True)
    # Short/simple LOSO folds first; multi-session people and later sessions follow.
    tags = ['loso_P'+f'{v:02}' for v in (1, 2, 3, 5, 6, 7, 8, 9, 4, 10)]+['later_last']
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fold_job, (t, str(args.cache), str(args.out), deadline)): t for t in tags}
        for future in as_completed(futures):
            tag = futures[future]
            try: future.result()
            except Exception as error:
                failures.append(dict(tag=tag, error=str(error), traceback=traceback.format_exc()))
                print('FOLD FAILED', tag, repr(error), flush=True)
            write_json(args.out/'failures.json', failures)
            summarize(args.out)
    summarize(args.out)
    write_json(args.out/'runtime.json', dict(seconds=time.monotonic()-start, failures=len(failures)))
    print('FINISHED', 'seconds', round(time.monotonic()-start, 1), 'failures', len(failures), flush=True)

if __name__ == '__main__': main()
