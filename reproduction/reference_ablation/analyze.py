"""Verify every candidate/control prediction and report the paired comparison."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "2026-competition"))
from benchmark_utils.metrics import balanced_accuracy, group_scores
from run import original_snapshot, sha256, write_json
from references import CONTROL, CANDIDATES

VARIANTS = (CONTROL,) + CANDIDATES
LABELS = {
    CONTROL: "Лог-евклидово, накопительное (контроль)",
    "euclid_cumulative": "Арифметическое, накопительное",
    "riemann_recursive": "Риманово, рекурсивное",
    "logeuclid_ema": "Лог-евклидово, забывание 0,95",
}
MODEL_LABELS = {"logreg": "Один классификатор", "forest12": "Лес, 12 каналов", "forest20": "Лес, 20 каналов"}


def main():
    manifest = json.loads((HERE / "experiment.json").read_text())
    if original_snapshot() != manifest["original_artifact_sha256"]:
        raise RuntimeError("Original control artifacts changed during the comparison")
    for name, digest in manifest["code_sha256"].items():
        if sha256(HERE / name) != digest:
            raise RuntimeError(f"Computation code changed: {name}")
    for s, digest in manifest["data_sha256"].items():
        if sha256(ROOT / "data" / f"proteus_{s}.npz") != digest:
            raise RuntimeError(f"Data changed: {s}")
    for repo, expected in [("Brain-Decoding", manifest["source_commit"]),
                            ("2026-competition", manifest["competition_commit"])]:
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / repo, text=True).strip()
        if actual != expected or subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT / repo):
            raise RuntimeError(f"Original source changed: {repo}")

    control_dir = ROOT / "results/eeg"
    expected_names = {p.name for p in control_dir.glob("*_predictions.npz")}
    if len(expected_names) != 29:
        raise RuntimeError("Control does not contain the expected 29 jobs")
    records, verified = [], []
    for variant in VARIANTS:
        directory = control_dir if variant == CONTROL else HERE / "results" / variant
        prediction_files = sorted(directory.glob("*_predictions.npz"))
        if {p.name for p in prediction_files} != expected_names:
            raise RuntimeError(f"{variant}: incomplete/extra jobs: {len(prediction_files)}/29")
        for path in prediction_files:
            meta_path = path.with_name(path.name.replace("_predictions.npz", ".json"))
            meta = json.loads(meta_path.read_text())
            control = json.loads((control_dir / meta_path.name).read_text())
            if variant != CONTROL:
                if meta["variant"] != variant or meta["reference_update"] != manifest["variants"][variant]:
                    raise AssertionError("Variant configuration mismatch")
            for key in ["train_subjects", "eval_subjects", "train_windows", "eval_windows", "model", "protocol", "fold"]:
                if meta[key] != control[key]:
                    raise AssertionError(f"{path}: control mismatch in {key}")
            if meta["model"] != "logreg" and meta["trees"] != control["trees"]:
                raise AssertionError("Random channel sets or bands changed")
            with np.load(path) as f, np.load(control_dir / path.name) as baseline:
                for key in ["y", "subject", "session", "context", "run", "onset"]:
                    if not np.array_equal(f[key], baseline[key]):
                        raise AssertionError(f"{path}: evaluation-window ordering changed")
                y, pred = f["y"], f["pred"]
                if pred.shape != y.shape or not set(np.unique(pred)) <= {0, 1, 2}:
                    raise AssertionError("Invalid predictions")
                stream = np.char.add(np.char.add(f["subject"], "/"), f["session"])
                cells = np.stack([stream, f["context"]], axis=1)
                metrics = dict(
                    session_mean=float(group_scores(balanced_accuracy, y, pred, stream).mean()),
                    cell_mean=float(group_scores(balanced_accuracy, y, pred, cells).mean()),
                    pooled=balanced_accuracy(y, pred), accuracy=float(np.mean(y == pred)))
                for key, value in metrics.items():
                    if not np.isclose(value, meta[key], atol=1e-14, rtol=0):
                        raise AssertionError(f"Official re-scoring differs: {path} / {key}")
            if meta["protocol"] in ("loso", "test") and set(meta["train_subjects"]) & set(meta["eval_subjects"]):
                raise AssertionError("Subject leakage")
            if variant != CONTROL:
                delta = metrics["session_mean"] - control["session_mean"]
                if not np.isclose(delta, meta["session_mean_delta_vs_control"], atol=1e-14, rtol=0):
                    raise AssertionError("Saved paired score difference is incorrect")
            records.append({k: v for k, v in dict(meta, variant=variant).items() if k != "trees"})
            verified.append({"variant": variant, "job": path.stem, "official_scores_match": True})
    frame = pd.DataFrame(records)
    frame.to_csv(HERE / "fold_scores.csv", index=False)
    rows, paired = [], []
    for variant in VARIANTS:
        for model in MODEL_LABELS:
            r = frame[(frame.variant == variant) & (frame.model == model)]
            loso = r[r.protocol == "loso"].set_index("fold").session_mean.sort_index()
            baseline_loso = frame[(frame.variant == CONTROL) & (frame.model == model) & (frame.protocol == "loso")].set_index("fold").session_mean.sort_index()
            if len(loso) != 8 or not loso.index.equals(baseline_loso.index):
                raise AssertionError("LOSO pairing is incomplete")
            delta = loso - baseline_loso
            later = r[r.protocol == "later"].session_mean
            test = r[r.protocol == "test"].session_mean
            if len(later) != 1 or len(test) != (0 if model == "forest12" else 1):
                raise AssertionError("Later/test protocol differs from the original")
            baseline_later = frame[(frame.variant == CONTROL) & (frame.model == model) & (frame.protocol == "later")].session_mean.iloc[0]
            baseline_test = frame[(frame.variant == CONTROL) & (frame.model == model) & (frame.protocol == "test")].session_mean
            row = dict(variant=variant, model=model, loso_mean=loso.mean(), loso_std=loso.std(),
                       delta_loso=delta.mean(), delta_loso_std=delta.std(),
                       loso_wins=int((delta > 1e-12).sum()), loso_ties=int((abs(delta) <= 1e-12).sum()),
                       loso_losses=int((delta < -1e-12).sum()), later=later.iloc[0],
                       delta_later=later.iloc[0]-baseline_later,
                       test=test.iloc[0] if len(test) else np.nan,
                       delta_test=test.iloc[0]-baseline_test.iloc[0] if len(test) else np.nan)
            rows.append(row)
            if variant != CONTROL:
                for subject in loso.index:
                    paired.append(dict(variant=variant, model=model, subject=subject,
                                       control=baseline_loso[subject], candidate=loso[subject],
                                       delta=delta[subject]))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(HERE / "comparison.csv", index=False)
    pd.DataFrame(paired).to_csv(HERE / "paired_loso.csv", index=False)
    verification = {
        "candidate_jobs": 87, "control_jobs": 29, "officially_rescored_files": len(verified),
        "all_scores_match_official_code": True, "metric_atol": 1e-14,
        "all_original_artifacts_unchanged": True, "source_checkouts_unchanged": True,
        "data_hashes_match_control": True, "all_tree_selections_match_control": True,
        "all_test_window_orders_match_control": True, "paired_loso_rows": len(paired),
        "source_commit": manifest["source_commit"], "competition_commit": manifest["competition_commit"],
    }
    write_json(HERE / "verification.json", verification)
    write_json(HERE / "verified_jobs.json", verified)
    make_plots(frame, comparison)
    make_report(comparison, verification, manifest)
    print(comparison.to_string(index=False))
    print(json.dumps(verification, indent=2))
    print(HERE / "REPORT.md")


def make_plots(frame, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    short = ["Control\nlog-Euclidean", "Euclidean\ncumulative", "Riemannian\nrecursive", "Log-Euclidean\nforgetting .95"]
    colors = ["#667085", "#0077b6", "#7b2cbf", "#e76f51"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, model in zip(axes.flat, MODEL_LABELS):
        subset = comparison[comparison.model == model].set_index("variant").loc[list(VARIANTS)]
        ax.bar(range(4), 100*subset.loso_mean, color=colors, alpha=.85,
               yerr=100*subset.loso_std, capsize=4)
        ax.axhline(100/3, color="gray", linestyle="--", linewidth=1)
        ax.set(xticks=range(4), xticklabels=short, ylim=(0, 85),
               ylabel="LOSO balanced accuracy (%)", title={"logreg": "One classifier", "forest12": "Forest: 12 channels", "forest20": "Forest: 20 channels"}[model])
        ax.tick_params(axis="x", labelsize=8)
    paired = pd.read_csv(HERE / "paired_loso.csv")
    ax = axes[1, 1]
    for variant, color in zip(CANDIDATES, colors[1:]):
        g = paired[(paired.model == "forest20") & (paired.variant == variant)].sort_values("subject")
        ax.plot(range(8), 100*g.delta, "o-", color=color, label=short[list(VARIANTS).index(variant)].replace("\n", " "))
    ax.axhline(0, color="gray", linewidth=1)
    ax.set(xticks=range(8), xticklabels=[s.rsplit("/", 1)[-1] for s in g.subject],
           ylabel="Change vs control (percentage points)", title="Forest 20: paired change per person")
    ax.legend(fontsize=8)
    fig.suptitle("Reference update comparison — same splits, seed, trees and classifiers\nError bars: sample SD across 8 subjects (not confidence intervals)", fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "comparison.png", dpi=170)
    plt.close(fig)


def make_report(comparison, verification, manifest):
    lines = [
        "# Сравнение опорных матриц на PROTEUS", "",
        "Повторён потоковый эксперимент таблицы 2: три новых способа обновления опорной матрицы, "
        "87 новых оценок и 29 ранее воспроизведённых контрольных оценок. Все предсказания независимо "
        "пересчитаны официальными метриками соревнования.", "",
        "## Результаты", "",
        "Сбалансированная точность, средняя по сессиям. LOSO: среднее ± выборочное SD по восьми людям. "
        "Δ — разность с тем же классификатором при исходном лог-евклидовом накопительном обновлении, "
        "в процентных пунктах. Плюс означает улучшение.", "",
    ]
    for model, label in MODEL_LABELS.items():
        lines += [f"### {label}", "",
                  "| Опорная матрица | LOSO | Δ LOSO, п.п. | Улучшение / равенство / ухудшение по людям | Поздняя сессия | Test |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in comparison[comparison.model == model].itertuples():
            test = "—" if pd.isna(r.test) else f"{r.test:.6f}"
            wins = "—" if r.variant == CONTROL else f"{r.loso_wins} / {r.loso_ties} / {r.loso_losses}"
            lines.append(f"| {LABELS[r.variant]} | {r.loso_mean:.6f} ± {r.loso_std:.6f} | {100*r.delta_loso:+.2f} | {wins} | {r.later:.6f} | {test} |")
        lines.append("")
    lines += ["![Сравнение и парные изменения](comparison.png)", "", "## Что показало сравнение", ""]
    for model, label in MODEL_LABELS.items():
        best = comparison[comparison.model == model].sort_values("loso_mean", ascending=False).iloc[0]
        lines.append(f"- {label}: наибольшее среднее LOSO у варианта «{LABELS[best.variant]}», "
                     f"{best.loso_mean:.6f}; Δ к контролю {100*best.delta_loso:+.2f} п.п.")
    forest = comparison[comparison.model == "forest20"].set_index("variant")
    lines += ["", "Для леса с 20 каналами изменения на публичном test:", ""]
    for variant in CANDIDATES:
        r = forest.loc[variant]
        lines.append(f"- {LABELS[variant]}: {r.test:.6f}, Δ {100*r.delta_test:+.2f} п.п.")
    lines += ["", "## Точные определения вариантов", "",
        "Для всех вариантов начальная матрица **одинакова**: M0=exp(L0), "
        "где L0 — среднее логарифмов обучающих ковариаций, как у автора. Вес prior=5. "
        "Меняется только правило сессионного обновления; это не сравнение разных оценок начального prior.", "",
        "1. **Контроль:** L_t=(5 L0+Σ log R_i)/(5+t); M_t=exp(L_t).",
        "2. **Арифметическое:** M_t=(5 M0+Σ R_i)/(5+t).",
        "3. **Риманово рекурсивное:** M_t=M_(t−1) #_(1/(5+t)) R_t, "
        "где A #_a B=A^(1/2)(A^(−1/2)BA^(−1/2))^a A^(1/2). "
        "Это индуктивная оценка, которая при некоммутирующих матрицах не равна точному среднему Фреше по всей истории.",
        "4. **Лог-евклидово с забыванием:** S_t=0.95 S_(t−1)+log R_t, "
        "n_t=0.95 n_(t−1)+1, L_t=S_t/n_t; S0=5 L0,n0=5. "
        "Это нормализованные экспоненциальные веса; шаг стремится к 0.05, но во время разогрева не постоянен. "
        "Период полураспада веса окна — 13.51 окна, а не фиксированное число секунд.", "",
        "Каждое текущее окно включается в опорную матрицу до его классификации, как в исходном коде. "
        "Новое правило применяется и к обучающим сессиям, и к предсказанию. Классификатор и StandardScaler "
        "переобучаются на соответствующих признаках, а их настройки сохраняются.", "",
        "## Что сохранено из исходного эксперимента", "",
        "- PROTEUS v1.0.0; те же npz-файлы с проверенными SHA-256, 41 канал, 480 отсчётов на окно.",
        "- Те же train/val/test: 1521/379/755 окон и 6/2/2 человека; те же восемь LOSO-разбиений и поздняя сессия.",
        "- Те же полосовые фильтры, OAS, определение касательных признаков, C=0.01, prior_weight=5 и seed=0.",
        "- Те же 50 деревьев и подмножества по 12/20 каналов; исходный код выбора полос, каналов и бутстрепа не переписан.",
        "- Совпадение списков полос и каналов каждого дерева с контролем проверено для всех лесов.",
        "- Потоковая подача одного окна, тот же порядок записей и сброс состояния в начале каждой сессии.",
        "- Для test обучение только на шести train-испытуемых; как в исходном протоколе, test есть только для logreg и forest20.", "",
        "Синтетические проверки статьи не изменялись: у них фиксированная единичная опорная точка и отсутствует "
        "исходное сессионное обновление. Подстановка нового обновления туда означала бы другой синтетический эксперимент.", "",
        "## Проверки и ограничения", "",
        f"- Официальным кодом заново оценены **{verification['officially_rescored_files']}** файлов предсказаний; "
        "все четыре метрики совпали с сохранёнными при абсолютном допуске 1e-14.",
        "- Исходные репозитории, подготовленные данные и все 29 исходных результатов остались неизменными.",
        "- Аналитические тесты проверяют матричные формулы, нормировку prior, затухание весов, "
        "положительную определённость, отсутствие влияния будущих окон и равенство пакетного/потокового ответа.",
        "- Контрольный адаптер даёт те же коэффициенты и ответы, что исходный код, на проверочном наборе.",
        "- Варианты и decay=0.95 зафиксированы до вычисления результатов; подбора скорости забывания по test не было.",
        "- Это исследовательское сравнение с одним seed, восемью dev-испытуемыми, одной поздней сессией и двумя "
        "public-test испытуемыми. Среднее улучшение не доказывает универсальное превосходство метода. "
        "SD на графике — разброс людей, не доверительный интервал; парные разности доступны отдельно.",
        "- Открытый test уже использовался в предыдущем воспроизведении; этот эксперимент не заменяет закрытую финальную оценку соревнования.", "",
        "## Файлы и повторный запуск", "",
        "- `README.md`: команды запуска и структура файлов.",
        "- `experiment.json`: зафиксированные формулы, хеши кода/данных/контроля и окружение.",
        "- `comparison.csv`: итоговые средние, SD и изменения к контролю.",
        "- `paired_loso.csv`: 72 парных изменения по испытуемым.",
        "- `fold_scores.csv`: все 116 оценок, включая контроль.",
        "- `results/<variant>/*_predictions.npz`: новые предсказания для каждого окна.",
        "- `verification.json`, `verified_jobs.json`: независимая проверка результатов.", "",
        "## Источники формул и кода", "",
        "- [pyRiemann: арифметическое среднее](https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.geometry.mean.mean_euclid.html).",
        "- [pyRiemann: аффинно-инвариантная геодезическая](https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.geometry.geodesic.geodesic_riemann.html).",
        f"- [Код автора](https://github.com/DorinDaniil/Brain-Decoding), commit `{manifest['source_commit']}`.",
        f"- [Код соревнования](https://github.com/neural-interfaces26/2026-competition), commit `{manifest['competition_commit']}`.",
        "- Предыдущий отчёт: `../REPORT.md`; исходный протокол — раздел 5 статьи в рабочей папке.", "",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
