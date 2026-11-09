"""Run the generated notebook's smoke command in a fresh interpreter."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def test_generated_smoke_runs_in_fresh_process(tmp_path):
    nb = json.loads((HERE / 'notebook/metric_wave1.ipynb').read_text())
    source = ''.join(nb['cells'][3]['source'])
    tree = ast.parse(source)
    smoke = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'smoke' for t in node.targets))
    # Exercise identical imports, alignment, weighting, fitting and checkpoints
    # on CPU locally; Kaggle runs this same command on CUDA.
    smoke = smoke.replace('cuda:0', 'cpu').replace('/tmp/metric-cuda-smoke', str(tmp_path / 'fit'))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-c', smoke], cwd=HERE, env=env,
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'fit/complete.json').exists()
