"""Repeat the verified EEG protocol with a new per-window SPD representation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
from reproduce_eeg import COMMIT, sha256, validate_splits
from temporal_features import CANDIDATES, DEFINITIONS, SOURCES, fit_model
from temporal_cache import MatrixCache,config_digest
from bci.data import concat, load_split
from bci.evaluation import later_sessions, leave_one_subject_out, predict_streams, score
from threadpoolctl import threadpool_limits

MODELS = ("logreg", "forest12", "forest20")


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def original_snapshot():
    return {p.name: sha256(p) for p in sorted((ROOT / "results/eeg").iterdir()) if p.is_file()}


def previous_snapshot():
    output = {}
    for name in ["reference_ablation", "literature_ablation", "representation_ablation"]:
        previous = ROOT / name
        files = list((previous / "results").glob("*/*"))
        files += [p for p in previous.iterdir() if p.is_file()]
        output.update({str(p.relative_to(ROOT)): sha256(p) for p in sorted(files) if p.is_file()})
    return output


def cache_inventory(variants=CANDIDATES):
    directory = HERE / "cache"
    config = json.loads((directory / "manifest.json").read_text())
    receipts = {}
    for representation in variants:
        for band in range(6):
            tag = f"{representation}_band{band}"
            record = json.loads((directory / f"{tag}.complete.json").read_text())
            if record["config_digest"] != config_digest(config):
                raise RuntimeError("Cache configuration mismatch")
            if sha256(directory / f"{tag}.npy") != record["sha256"]:
                raise RuntimeError("Matrix cache changed")
            receipts[tag] = record
    return {"manifest_sha256": sha256(directory / "manifest.json"), "receipts": receipts}


def initialize(parts):
    source = ROOT / "Brain-Decoding"
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != COMMIT:
        raise RuntimeError("Author source commit changed")
    if subprocess.check_output(["git", "diff", "HEAD"], cwd=source):
        raise RuntimeError("Author source has tracked modifications")
    competition = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "2026-competition", text=True).strip()
    if competition != "2b53aec91c111f8b5423b6674f096c48cc1a99af":
        raise RuntimeError("Competition source commit changed")
    if subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT / "2026-competition"):
        raise RuntimeError("Competition source has tracked modifications")
    baseline = json.loads((ROOT / "results/eeg/manifest.json").read_text())
    hashes = {s: sha256(ROOT / "data" / f"proteus_{s}.npz") for s in parts}
    if hashes != baseline["data_sha256"]:
        raise RuntimeError("Prepared data differ from verified control")
    versions = {p: importlib.metadata.version(p) for p in baseline["package_versions"]}
    if versions != baseline["package_versions"]:
        raise RuntimeError("Package versions differ from verified control")
    summary = validate_splits(parts)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": COMMIT, "competition_commit": competition,
        "data_sha256": hashes, "package_versions": versions, "splits": summary,
        "original_artifact_sha256": original_snapshot(),
        "previous_artifact_sha256": previous_snapshot(),
        "code_sha256": {"temporal_features.py": sha256(HERE / "temporal_features.py"), "temporal_cache.py": sha256(HERE / "temporal_cache.py"), "run.py": sha256(Path(__file__)), "select_model.py":sha256(HERE/"select_model.py")},
        "cache": {"manifest_sha256": sha256(HERE / "cache/manifest.json")},
        "variants": DEFINITIONS, "parameters": baseline["parameters"],
        "representation_parameters": {"locked_protocol_sha256":sha256(HERE/"PROTOCOL.md"), "lag4":4,"pyramid_halves":2,"signature_depth":2,"signature_patches":16,"signature_weight":1,"path_quadrature_nodes":2},
        "prior": "Original mean(log(R_train)), fitted separately to the new R; weight=5; original cumulative log-Euclidean updater unchanged",
        "protocol": "29 jobs per candidate: 24 LOSO, 3 later, 2 test; train-only fit for public test",
        "test_policy": "Six literature-driven fixed SPD representations; select on forest20 LOSO and freeze selection before any new test; no test tuning",
        "geometry_note": "Only R(X) changes; inherited original SessionRecentering; delay copies are grouped by physical electrode",
        "attribution": "Published mechanisms with explicit per-window adaptations; see PROTOCOL.md and literature/survey.md",
        "sources": SOURCES,
    }
    master_file = HERE / "experiment.json"
    if master_file.exists():
        previous = json.loads(master_file.read_text())
        for key in manifest:
            if key != "created_utc" and previous[key] != manifest[key]:
                raise RuntimeError(f"Experiment cannot resume with changed {key}")
    else:
        write_json(master_file, manifest)
    for variant in CANDIDATES:
        out = HERE / "results" / variant
        out.mkdir(parents=True, exist_ok=True)
    print("Validated exact source/data/environment and frozen experiment manifest", flush=True)


def run_job(variant, name, protocol, fold, train, test, cache, cache_hashes):
    tag = f"{name}_{protocol}_{fold.replace('/', '_')}"
    out = HERE / "results" / variant
    result_file = out / f"{tag}.json"
    prediction_file = out / f"{tag}_predictions.npz"
    if result_file.exists():
        if not prediction_file.exists():
            raise RuntimeError(f"Missing predictions for resumed job {variant}/{tag}")
        if json.loads(result_file.read_text())["matrix_cache_sha256"] != cache_hashes:
            raise RuntimeError("Cached matrices changed for a completed model job")
        print(f"RESUME {variant}/{tag}", flush=True)
        return
    control = json.loads((ROOT / "results/eeg" / f"{tag}.json").read_text())
    if sorted(set(train.subject)) != control["train_subjects"] or sorted(set(test.subject)) != control["eval_subjects"]:
        raise RuntimeError("Subject fold differs from original control")
    with np.load(ROOT / "results/eeg" / f"{tag}_predictions.npz") as f:
        for key in ["y", "subject", "session", "context", "run", "onset"]:
            if not np.array_equal(getattr(test, key), f[key]):
                raise RuntimeError(f"Evaluation-window {key} differ from original control")
    print(f"START {variant}/{tag} train={len(train)} eval={len(test)}", flush=True)
    start = time.monotonic()
    with threadpool_limits(limits=1):
        model = fit_model(train, name, variant, cache=cache)
        fit_seconds = time.monotonic() - start
        tree_specs = None
        if hasattr(model, "trees"):
            tree_specs = [{"band": t.band, "channels": getattr(t,"physical_channels",t.channels).tolist()} for t in model.trees]
            if tree_specs != control["trees"]:
                raise RuntimeError("Random bands/channel subsets differ from original control")
        prediction = predict_streams(model, test)
    metrics = score(test, prediction)
    result = {
        "variant": variant, "model": name, "protocol": protocol, "fold": fold,
        "train_subjects": sorted(set(train.subject)), "eval_subjects": sorted(set(test.subject)),
        "train_windows": len(train), "eval_windows": len(test),
        "fit_seconds": fit_seconds, "total_seconds": time.monotonic() - start,
        "session_mean_delta_vs_control": metrics["session_mean"] - control["session_mean"],
        "representation_definition": DEFINITIONS[variant], "reference_update": "original cumulative log-Euclidean, prior_weight=5", **metrics,
        "matrix_cache_sha256": cache_hashes,
    }
    if tree_specs is not None:
        result["trees"] = tree_specs
        result["matrix_channels"] = [t.channels.tolist() for t in model.trees]
        result["tangent_dimension"] = int(model.trees[0].clf.n_features_in_)
    else:
        result["tangent_dimension"] = int(model.clf.n_features_in_)
    np.savez_compressed(prediction_file, y=test.y, pred=prediction, subject=test.subject,
                        session=test.session, context=test.context, run=test.run, onset=test.onset)
    write_json(result_file, result)
    print(f"DONE {variant}/{tag}: {metrics['session_mean']:.9f}, "
          f"delta={result['session_mean_delta_vs_control']:+.9f} ({result['total_seconds']:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["all", *CANDIDATES], default="all")
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--protocol", choices=["all", "cv", "test"], default="all")
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    parts = {s: load_split(s, ROOT / "data") for s in ["train", "val", "test"]}
    initialize(parts)
    if args.initialize:
        return
    if args.protocol in ("all","test"):
        if not (HERE/"selection.json").exists():
            raise RuntimeError("Freeze LOSO selection using select.py before any test evaluation")
        from select_model import validate_selection
        validate_selection()
    cache = MatrixCache(HERE / "cache")
    dev = concat(parts["train"], parts["val"])
    variants = CANDIDATES if args.variant == "all" else [args.variant]
    for variant in variants:
        inventory = cache_inventory([variant])
        cache_hashes = {tag: r["sha256"] for tag,r in inventory["receipts"].items()}
        for name in MODELS:
            if args.model != "all" and args.model != name:
                continue
            if args.protocol in ("all", "cv"):
                for fold, train, test in leave_one_subject_out(dev):
                    run_job(variant, name, "loso", str(fold), train, test, cache, cache_hashes)
                for _, train, test in later_sessions(dev):
                    if len(set(test.stream)) != 1 or len(set(test.subject)) != 1:
                        raise RuntimeError("Later-session fold changed")
                    run_job(variant, name, "later", "last", train, test, cache, cache_hashes)
            if args.protocol in ("all", "test") and name in ("logreg", "forest20"):
                run_job(variant, name, "test", "official", parts["train"], parts["test"], cache, cache_hashes)


if __name__ == "__main__":
    main()
