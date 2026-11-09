"""Execute paper table 2 with the unmodified upstream Track 2 implementation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "8d577af42619e5911cc5da8743c8a3e9b629014b"
PAPER = {
    "logreg": {"loso_mean": .469, "loso_std": .114, "later": .547, "test": .455},
    "forest12": {"loso_mean": .511, "loso_std": .133, "later": .614},
    "forest20": {"loso_mean": .531, "loso_std": .150, "later": .669, "test": .529},
}


def validate_splits(parts, check_paper_counts=True):
    expected = {"train": (1521, 6), "val": (379, 2), "test": (755, 2)}
    summary = {}
    for name, w in parts.items():
        if w.X.shape != (len(w.y), 41, 480) or not np.isfinite(w.X).all():
            raise ValueError(f"{name}: invalid EEG shape or non-finite samples")
        if set(np.unique(w.y)) != {0, 1, 2}:
            raise ValueError(f"{name}: expected all three classes")
        order = np.lexsort((w.onset, w.run, w.session, w.subject))
        if not np.array_equal(order, np.arange(len(w.y))):
            raise ValueError(f"{name}: windows are not in recording order")
        summary[name] = {"windows": len(w.y), "subjects": len(set(w.subject)),
                         "subject_ids": sorted(set(w.subject)),
                         "sessions": len(set(zip(w.subject, w.session)))}
        if check_paper_counts and (len(w.y), len(set(w.subject))) != expected[name]:
            raise ValueError(f"{name}: differs from paper counts: {summary[name]}")
    for i, a in enumerate(parts):
        for b in list(parts)[i + 1:]:
            if set(parts[a].subject) & set(parts[b].subject):
                raise ValueError(f"subject overlap: {a} / {b}")
            if not np.array_equal(parts[a].ch_names, parts[b].ch_names):
                raise ValueError(f"channel ordering mismatch: {a} / {b}")
    return summary


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "Brain-Decoding")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "results/eeg")
    parser.add_argument("--protocol", choices=["cv", "test", "all"], default="all")
    parser.add_argument("--model", choices=["logreg", "forest12", "forest20", "all"], default="all")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != COMMIT or subprocess.check_output(["git", "diff", "HEAD"], cwd=source):
        raise RuntimeError("Upstream source must be unchanged at the recorded commit")
    sys.path.insert(0, str(source / "track2_bci"))
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from bci.data import concat, load_split
    from bci.evaluation import later_sessions, leave_one_subject_out, predict_streams, score
    from bci.models import forest, logreg

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    parts = {s: load_split(s, args.data_dir) for s in ["train", "val", "test"]}
    summary = validate_splits(parts)
    manifest = {
        "source_commit": commit, "source_url": "https://github.com/DorinDaniil/Brain-Decoding",
        "created_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "python": sys.version, "splits": summary,
        "data_sha256": {s: sha256(args.data_dir / f"proteus_{s}.npz") for s in parts},
        "package_versions": {p: importlib.metadata.version(p) for p in
            ["numpy", "scipy", "scikit-learn", "pyriemann", "mne", "neuralbench", "pandas", "torch"]},
        "parameters": {"C": .01, "seed": 0, "prior_weight": 5, "n_trees": 50,
                       "forest_channels": [12, 20], "bands": forest.BANDS},
        "test_training": "train only (6 subjects), as upstream notebooks",
        "reference": PAPER,
    }
    manifest_file = out / "manifest.json"
    if manifest_file.exists():
        previous = json.loads(manifest_file.read_text())
        for key in ["source_commit", "data_sha256", "package_versions", "parameters"]:
            # JSON round-trip normalizes band tuples.
            if previous[key] != json.loads(json.dumps(manifest[key])):
                raise RuntimeError(f"Cannot resume with changed {key}")
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if args.validate_only:
        return
    fits = {"logreg": lambda tr: logreg.fit(tr, C=.01, prior_weight=5, seed=0),
            "forest12": lambda tr: forest.fit(tr, n_trees=50, n_channels=12, C=.01, prior_weight=5, seed=0),
            "forest20": lambda tr: forest.fit(tr, n_trees=50, n_channels=20, C=.01, prior_weight=5, seed=0)}
    dev = concat(parts["train"], parts["val"])

    def run_job(model_name, protocol, fold, train, test):
        safe_fold = fold.replace("/", "_")
        tag = f"{model_name}_{protocol}_{safe_fold}"
        result_file = out / f"{tag}.json"
        if result_file.exists():
            print(f"RESUME {tag}", flush=True)
            return
        print(f"START {tag} train={len(train)} eval={len(test)}", flush=True)
        start = time.monotonic()
        with threadpool_limits(limits=1):
            model = fits[model_name](train)
            fit_seconds = time.monotonic() - start
            pred = predict_streams(model, test)
        scores = score(test, pred)
        result = {"model": model_name, "protocol": protocol, "fold": fold,
                  "train_subjects": sorted(set(train.subject)), "eval_subjects": sorted(set(test.subject)),
                  "train_windows": len(train), "eval_windows": len(test),
                  "fit_seconds": fit_seconds, "total_seconds": time.monotonic() - start,
                  **scores}
        # Preserve full per-window outputs so every reported metric is independently checkable.
        np.savez_compressed(out / f"{tag}_predictions.npz", y=test.y, pred=pred,
                            subject=test.subject, session=test.session, context=test.context,
                            run=test.run, onset=test.onset)
        if hasattr(model, "trees"):
            result["trees"] = [{"band": t.band, "channels": t.channels.tolist()} for t in model.trees]
        result_file.write_text(json.dumps(result, indent=2))
        print(f"DONE {tag}: session_mean={scores['session_mean']:.9f} ({result['total_seconds']:.1f}s)", flush=True)

    if args.protocol in ["cv", "all"]:
        for name in fits:
            if args.model != "all" and name != args.model:
                continue
            for fold, tr, te in leave_one_subject_out(dev):
                run_job(name, "loso", str(fold), tr, te)
            for _, tr, te in later_sessions(dev):
                if len(set(te.subject)) != 1 or len(set(te.stream)) != 1:
                    raise ValueError("Later-session fold differs from paper (one subject/session)")
                run_job(name, "later", "last", tr, te)
    if args.protocol in ["test", "all"]:
        for name in ["logreg", "forest20"]:
            if args.model != "all" and name != args.model:
                continue
            run_job(name, "test", "official", parts["train"], parts["test"])

    records = [json.loads(p.read_text()) for name in fits for p in out.glob(f"{name}_*.json")]
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "trees"} for r in records])
    frame.to_csv(out / "fold_scores.csv", index=False)
    rows = []
    for model in fits:
        row = {"model": model}
        subset = frame[frame.model == model]
        loso = subset.loc[subset.protocol == "loso", "session_mean"]
        row.update(loso_folds=len(loso), loso_mean=loso.mean(), loso_std=loso.std())
        for protocol in ["later", "test"]:
            values = subset.loc[subset.protocol == protocol, "session_mean"]
            row[protocol] = values.iloc[0] if len(values) else None
        for metric, reference in PAPER[model].items():
            row[f"paper_{metric}"] = reference
            row[f"delta_{metric}"] = None if row.get(metric) is None else row[metric] - reference
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "comparison.csv", index=False)
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
