"""Paired summary of results.csv (steps 1-2) and tta_results.csv (step 3), session_mean."""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LATER = "later sessions"


def paired(df, key, base, value="session_mean"):
    """Per protocol: mean, diff vs base over paired (seed, fold) runs, se of the diff
    (LOSO: per-subject diffs averaged over seeds; later: per-seed diffs), wins."""
    rows = []
    for proto, d in [("LOSO", df[df.fold != LATER]), ("later", df[df.fold == LATER])]:
        wide = d.pivot_table(index=["seed", "fold"], columns=key, values=value)
        wide = wide.dropna(subset=[base])
        for c in wide.columns:
            w = wide[[base]].dropna() if c == base else wide[[base, c]].dropna()
            if w.empty:
                continue
            diff = (w[c] - w[base])
            unit = diff.groupby(level="fold" if proto == "LOSO" else "seed").mean()
            rows.append(dict(protocol=proto, config=c, n_runs=len(w), mean=w[c].mean(),
                             diff=diff.mean(), se=unit.std(ddof=1) / np.sqrt(len(unit)) if len(unit) > 1 else np.nan,
                             better=f"{int((unit > 0).sum())}/{len(unit)}"))
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    r = pd.read_csv(f"{HERE}/results.csv")
    r = r[~r.fold.str.endswith("#recheck")]
    print("steps 1-2 (vs lag100; LOSO 'better' = subjects, later = seeds)")
    print(paired(r, "config", "lag100").to_string(index=False))
    if os.path.exists(f"{HERE}/tta_results.csv"):
        t = pd.read_csv(f"{HERE}/tta_results.csv")
        for name, d in [("seed 0 (selection)", t[t.seed == 0]), ("seeds 1-3 LOSO + later 0-5", t[(t.seed != 0) | (t.fold == LATER)])]:
            print(f"\nstep 3 TTA on lag100, {name}")
            print(paired(d, "variant", "none").to_string(index=False))
