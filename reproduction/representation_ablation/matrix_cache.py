"""Exact, read-only per-window SPD cache; no learned or label-dependent state."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from representations import CANDIDATES, compute_matrices

HERE = Path(__file__).resolve().parent


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    temporary = Path(str(path)+".tmp")
    temporary.write_text(json.dumps(obj, indent=2))
    temporary.replace(path)


def window_key(X):
    if np.asarray(X).dtype != np.float32:
        raise TypeError("Exact cache expects original float32 windows")
    return hashlib.blake2b(np.ascontiguousarray(X).tobytes(), digest_size=20).hexdigest()


def initialize_cache(directory, X, bands, data_sha256):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if X.dtype != np.float32 or X.ndim != 3:
        raise ValueError("Cache input must be raw prepared float32 (N,C,T)")
    config = {
        "shape": list(X.shape), "dtype": "float32", "bands": [list(b) for b in bands],
        "data_sha256": data_sha256,
        "window_keys": [window_key(x) for x in X],
        "representation_code_sha256": file_sha(HERE / "representations.py"),
        "cache_code_sha256": file_sha(Path(__file__)),
        "representations": list(CANDIDATES),
        "label_policy": "Pure per-window transforms; no labels or cross-window statistics",
        "exact_hsic": "All time samples, float64 kernels, exact symmetric Gram, no approximation",
    }
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != config:
            raise RuntimeError("Cache input/configuration/code changed")
    else:
        write_json(manifest_path, config)
    return config


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def build_one(directory, X, representation, band_index, chunk_size=16):
    directory = Path(directory)
    config = json.loads((directory / "manifest.json").read_text())
    if config["shape"] != list(X.shape) or [window_key(x) for x in X] != config["window_keys"]:
        raise RuntimeError("Input windows do not match the fixed cache index")
    if file_sha(HERE / "representations.py") != config["representation_code_sha256"] or file_sha(Path(__file__)) != config["cache_code_sha256"]:
        raise RuntimeError("Computation code changed during caching")
    if representation not in CANDIDATES:
        raise ValueError(representation)
    digest = config_digest(config)
    tag = f"{representation}_band{band_index}"
    final = directory / f"{tag}.npy"
    complete = directory / f"{tag}.complete.json"
    if complete.exists():
        record = json.loads(complete.read_text())
        if record["config_digest"] != digest or file_sha(final) != record["sha256"]:
            raise RuntimeError(f"Completed cache was changed: {tag}")
        print(f"RESUME cache {tag}", flush=True)
        return record
    partial = directory / f"{tag}.partial.npy"
    progress_file = directory / f"{tag}.progress.json"
    n, physical_channels, _ = X.shape
    d = physical_channels * (2 if representation == "delay1_oas" else 1)
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        if progress["config_digest"] != digest:
            raise RuntimeError("Partial cache configuration changed")
        start = progress["completed"]
        existing = partial if partial.exists() else final
        if not partial.exists() and start != n:
            raise RuntimeError("Incomplete cache lost its partial matrix file")
        matrices = np.load(existing, mmap_mode="r+")
        min_eigenvalue = progress["min_eigenvalue"]
        seconds = progress["compute_seconds"]
    else:
        matrices = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float64, shape=(n,d,d))
        start, min_eigenvalue, seconds = 0, float("inf"), 0.
    if matrices.shape != (n,d,d) or matrices.dtype != np.float64:
        raise RuntimeError("Invalid matrix cache shape/dtype")
    begin = time.monotonic()
    for first in range(start,n,chunk_size):
        stop = min(n,first+chunk_size)
        block = compute_matrices(X[first:stop],tuple(config["bands"][band_index]),representation)
        if not np.isfinite(block).all() or not np.allclose(block,np.swapaxes(block,-1,-2),atol=1e-10,rtol=1e-10):
            raise RuntimeError(f"Invalid SPD matrices in {tag} at row {first}")
        eigen_min = float(np.linalg.eigvalsh(block).min())
        if eigen_min <= 0:
            raise RuntimeError(f"Non-positive matrix in {tag} at row {first}: {eigen_min}")
        min_eigenvalue = min(min_eigenvalue,eigen_min)
        matrices[first:stop] = block
        matrices.flush()
        progress = {"config_digest":digest,"completed":stop,"min_eigenvalue":min_eigenvalue,
                    "compute_seconds":seconds+time.monotonic()-begin}
        write_json(progress_file,progress)
        if stop%128==0 or stop==n:
            print(f"CACHE {tag}: {stop}/{n}",flush=True)
    del matrices
    if partial.exists():
        partial.replace(final)
    record = {"representation":representation,"band_index":band_index,"band":config["bands"][band_index],
              "shape":[n,d,d],"dtype":"float64","config_digest":digest,
              "sha256":file_sha(final),"min_eigenvalue":min_eigenvalue,
              "compute_seconds":seconds+time.monotonic()-begin,
              "completed_utc":datetime.now(timezone.utc).isoformat()}
    write_json(complete,record)
    print(f"DONE cache {tag}: {record['compute_seconds']:.1f}s",flush=True)
    return record


class MatrixCache:
    """An immutable data source shared by model copies; session state is elsewhere."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "manifest.json").read_text())
        if file_sha(HERE / "representations.py") != self.config["representation_code_sha256"] or file_sha(Path(__file__)) != self.config["cache_code_sha256"]:
            raise RuntimeError("Cache computation code changed")
        self._index = {key:i for i,key in enumerate(self.config["window_keys"])}
        self._bands = {tuple(b):i for i,b in enumerate(self.config["bands"])}
        self._arrays = {}

    def __deepcopy__(self,memo):
        memo[id(self)] = self
        return self

    def lookup(self,representation,band,X):
        X = np.asarray(X)
        if X.dtype != np.float32:
            raise TypeError("Exact cache expects original float32 windows")
        if X.ndim!=3 or list(X.shape[1:])!=self.config["shape"][1:]:
            raise ValueError("Cache window shape mismatch")
        ids = [self._index[window_key(x)] for x in X]
        band_index = self._bands[tuple(band)]
        key = representation,band_index
        if key not in self._arrays:
            tag = f"{representation}_band{band_index}"
            if not (self.directory/f"{tag}.complete.json").exists():
                raise RuntimeError(f"Cache is incomplete: {tag}")
            receipt = json.loads((self.directory/f"{tag}.complete.json").read_text())
            if receipt["config_digest"] != config_digest(self.config):
                raise RuntimeError(f"Cache receipt has another configuration: {tag}")
            self._arrays[key] = np.load(self.directory/f"{tag}.npy",mmap_mode="r")
        return np.asarray(self._arrays[key][ids])
