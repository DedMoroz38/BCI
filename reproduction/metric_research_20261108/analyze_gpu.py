"""Verify all GPU fits and compare fixed mixtures on matching all10 OOF data."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'src'))
from common import ROOT, sha, write_json, softmax, balance, mix


def load(path):
    with np.load(path) as f: return {k: f[k].copy() for k in f.files}


def main():
    folder = HERE / 'downloads/gpu-v2'
    root = folder / 'metric-results'; out = HERE / 'to_human'
    manifest = json.loads((root / 'study-manifest.json').read_text())
    receipt = json.loads((folder / 'download-receipt.json').read_text())
    for name, digest in receipt['files'].items(): assert sha(folder / name) == digest
    for name, digest in manifest['code'].items(): assert sha(HERE / name) == digest
    for split, digest in manifest['data'].items(): assert sha(ROOT / 'data' / f'proteus_{split}.npz') == digest
    provenance = json.loads((root / 'bundle-provenance.json').read_text())
    build = json.loads((HERE / 'notebook-build.json').read_text())
    assert provenance['metric_bundle_sha256'] == build['bundle_sha256']
    execution = json.loads((root / 'execution.json').read_text())
    assert not execution['budget_pending'] and sum(j['done'] for j in execution['jobs']) == 55
    spec = importlib.util.spec_from_file_location('independent', ROOT / 'ensemble_comparison/analyze.py')
    independent = importlib.util.module_from_spec(spec); spec.loader.exec_module(independent)
    score = independent.independent_scores
    baseroot = HERE / 'experiments/all10_baseline/results'
    tags = sorted(p.name for p in baseroot.iterdir() if p.is_dir())
    assert len(tags) == 11
    base = {}
    for tag in tags:
        complete = json.loads((baseroot / tag / 'complete.json').read_text())
        for name, digest in complete['files'].items(): assert sha(baseroot / tag / name) == digest
        base[tag] = load(baseroot / tag / 'lag4.npz')
    records = {}; curves = []; fitrows = []; nll_deltas = []; softmax_deltas = []
    for path in sorted((root / 'models').glob('*/*/seed*/prediction.json')):
        r = json.loads(path.read_text()); parent = path.parent; tag = r['tag']; label = f"{r['config']}_s{r['seed']}"
        for name, key in [('prediction.npz','prediction_sha256'), ('model.pt','model_sha256'),
                          ('alignment.npz','alignment_sha256'), ('logits.npy','logits_sha256')]:
            assert sha(parent / name) == r[key]
        assert r['study_sha256'] == sha(root / 'study-manifest.json')
        data = load(parent / 'prediction.npz'); z = np.load(parent / 'logits.npy')
        for key in ('y','subject','session','context','run','onset'):
            np.testing.assert_array_equal(data[key], base[tag][key])
        np.testing.assert_array_equal(data['pred'], data['proba'].argmax(1))
        recomputed = softmax(z, axis=1)
        np.testing.assert_allclose(recomputed, data['proba'], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(recomputed.argmax(1), data['pred'])
        softmax_deltas.append(float(np.max(np.abs(recomputed-data['proba']))))
        for key, value in score(data).items(): np.testing.assert_allclose(value, r[key], rtol=0, atol=1e-12)
        ll = -np.log(np.maximum(data['proba'][np.arange(len(data['y'])), data['y']], 1e-15)).mean()
        brier = np.mean(np.sum((data['proba'] - np.eye(3)[data['y']])**2, axis=1))
        # GPU notebook/Linux and local ARM use float32 log/reduction kernels;
        # BA/labels remain exact, whereas NLL can differ by a few float32 ULPs.
        np.testing.assert_allclose(ll, r['log_loss'], rtol=0, atol=1e-6)
        np.testing.assert_allclose(brier, r['brier'], rtol=0, atol=1e-12)
        nll_deltas.append(abs(float(ll) - r['log_loss']))
        a = set(r['inner_train_people']); b = set(r['inner_val_people'])
        assert not a & b and a | b == set(r['train_people'])
        if tag.startswith('loso_'): assert not set(data['subject']) & set(r['train_people'])
        selection = json.loads((parent / 'selection/complete.json').read_text())
        refit = json.loads((parent / 'refit/complete.json').read_text())
        assert selection['best_epoch'] == r['best_epoch'] == refit['settings']['epochs'] == refit['stopped_epoch']
        history = pd.read_csv(parent / 'selection/curves.csv')
        refit_history = pd.read_csv(parent / 'refit/curves.csv')
        assert len(history) == selection['stopped_epoch'] and len(refit_history) == r['best_epoch']
        best = float('inf'); chosen = 0
        for row in history.itertuples():
            if row.val_cell_nll < best - 1e-4: best = row.val_cell_nll; chosen = row.epoch
        assert chosen == r['best_epoch']
        first = history.iloc[0]; last = history.iloc[-1]; selected = history[history.epoch == chosen].iloc[0]
        curves.append(dict(config=r['config'], seed=r['seed'], tag=tag, best_epoch=chosen,
            stopped_epoch=len(history), train_ce_first=first.train_ce, train_ce_last=last.train_ce,
            val_ce_first=first.val_ce, val_ce_last=last.val_ce,
            train_ba_last=last.train_cell_ba, val_ba_last=last.val_cell_ba,
            val_ba_selected=selected.val_cell_ba, val_ba_max=history.val_cell_ba.max(),
            best_ba_epoch=int(history.loc[history.val_cell_ba.idxmax(),'epoch']),
            nll_rises_train_falls=bool(last.train_ce < first.train_ce and last.val_ce > first.val_ce)))
        records.setdefault(label, {})[tag] = dict(data=data, logits=[z], folder=str(parent))
        fitrows.append(r)
    assert len(fitrows) == 55 and len(records) == 5
    for fits in records.values(): assert sorted(fits) == tags
    for config in ('shallow16_class','shallow16_subject'):
        records[config + '_mean2'] = {tag: dict(data=records[config+'_s0'][tag]['data'],
            logits=[records[config+'_s0'][tag]['logits'][0], records[config+'_s1'][tag]['logits'][0]]) for tag in tags}
    predictions = {'lag4': {tag: base[tag]['proba'] for tag in tags},
                   'lag4_balance1': {tag: load(baseroot/tag/'lag4_balance1.npz')['proba'] for tag in tags}}
    for label, fits in records.items():
        for kind in ('cnn','mix_T4','mix_T4_balance1'):
            values = {}
            for tag, fit in fits.items():
                temp = 1 if kind == 'cnn' else 4
                q = np.mean([softmax(z/temp, axis=1) for z in fit['logits']], axis=0)
                p = q if kind == 'cnn' else mix(base[tag]['proba'], q, .3)
                if kind.endswith('balance1'):
                    stream = np.char.add(np.char.add(base[tag]['subject'], '/'), base[tag]['session'])
                    p = balance(p, [np.flatnonzero(stream == s) for s in np.unique(stream)], 1)
                values[tag] = p
            predictions[label + '__' + kind] = values
    summary = []; perperson = []; cvtags = [tag for tag in tags if tag.startswith('loso_')]
    assert len(cvtags) == 10
    for label, values in predictions.items():
        parts = []; personal = []
        for tag in tags:
            data = dict(base[tag], proba=values[tag], pred=values[tag].argmax(1)); metrics = score(data)
            if tag in cvtags:
                parts.append(data); personal.append(metrics['cell_mean'])
                perperson.append(dict(variant=label, tag=tag, **metrics))
            else: late = metrics['cell_mean']
        merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        assert len(merged['y']) == 2655
        assert len(set(zip(merged['subject'],merged['session'],merged['run'],merged['onset']))) == 2655
        metrics = score(merged)
        nll = float(-np.log(np.maximum(merged['proba'][np.arange(2655),merged['y']],1e-15)).mean())
        summary.append(dict(variant=label, official_oof_cell=metrics['cell_mean'],
            mean_person_cell=float(np.mean(personal)), later_cell=late, log_loss=nll))
    df = pd.DataFrame(summary).sort_values('official_oof_cell', ascending=False)
    df.to_csv(out / 'GPU_ALL10_COMPARISON.csv', index=False)
    pd.DataFrame(perperson).to_csv(out / 'GPU_PERSON_SCORES.csv', index=False)
    pd.DataFrame(curves).to_csv(out / 'GPU_LEARNING_DIAGNOSTICS.csv', index=False)
    pd.DataFrame(fitrows).to_csv(out / 'GPU_VERIFIED_FITS.csv', index=False)
    replay_path = out / 'GPU_CPU_REPLAY.json'
    replay = json.loads(replay_path.read_text()) if replay_path.exists() else None
    if replay:
        assert replay['models'] == 55 and replay['labels_exact']
        assert replay['max_probability_delta'] <= 1e-4
    write_json(out / 'GPU_VERIFICATION.json', dict(fits=55, complete_config_seeds=5, held_out_people=10,
        completed_later_folds=5, download_files=len(receipt['files']), data_and_code_hashes=True,
        predictions_and_artifact_hashes=True, matching_metadata=True, independent_rescore=True,
        inner_outer_person_separation=True, epoch_selection_and_refit=True,
        bundle_sha256=build['bundle_sha256'], execution_seconds=execution['seconds'],
        max_float32_nll_rescore_delta=max(nll_deltas),
        max_float32_softmax_rescore_delta=max(softmax_deltas),
        cpu_model_replay='all55_passed' if replay else 'pending_separate_check',
        cpu_replay_receipt_sha256=sha(replay_path) if replay else None))
    print(df.to_string(index=False), flush=True)
    c = pd.DataFrame(curves)
    print('\nLEARNING DIAGNOSTICS\n', c.groupby(['config','seed']).agg(
        median_best_epoch=('best_epoch','median'), median_stop_epoch=('stopped_epoch','median'),
        mean_train_ce_last=('train_ce_last','mean'), mean_val_ce_last=('val_ce_last','mean'),
        mean_train_ba_last=('train_ba_last','mean'), mean_val_ba_last=('val_ba_last','mean'),
        overfit_curves=('nll_rises_train_falls','sum')).to_string(), flush=True)


if __name__ == '__main__': main()
