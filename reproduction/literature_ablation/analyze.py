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
from run import original_snapshot, previous_snapshot, sha256, write_json
from literature_references import CONTROL, CANDIDATES, DEFINITIONS, SOURCES

PREVIOUS = ("euclid_cumulative", "riemann_recursive", "logeuclid_ema")
VARIANTS = (CONTROL,) + PREVIOUS + CANDIDATES
DISPLAY_VARIANTS = (CONTROL,) + CANDIDATES
LABELS = {
    CONTROL: "Лог-евклидово, накопительное (контроль)",
    "euclid_cumulative": "Арифметическое, накопительное",
    "riemann_recursive": "Риманово, рекурсивное",
    "logeuclid_ema": "Лог-евклидово, забывание 0,95",
    "harmonic_cumulative": "Гармоническое",
    "poweuclid_half": "Степенное евклидово, p=1/2",
    "kullback_sym": "Симметричная KL / Jeffreys",
    "logchol_cumulative": "Log-Cholesky",
    "wasserstein_recursive": "Bures–Wasserstein, рекурсивное",
}
MODEL_LABELS = {"logreg": "Один классификатор", "forest12": "Лес, 12 каналов", "forest20": "Лес, 20 каналов"}


def main():
    manifest = json.loads((HERE / "experiment.json").read_text())
    if original_snapshot() != manifest["original_artifact_sha256"]:
        raise RuntimeError("Original control artifacts changed during the comparison")
    if previous_snapshot() != manifest["previous_artifact_sha256"]:
        raise RuntimeError("Previous three-variant experiment changed")
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
    previous_definitions = json.loads((ROOT / "reference_ablation/experiment.json").read_text())["variants"]
    records, verified = [], []
    for variant in VARIANTS:
        directory = (control_dir if variant == CONTROL else
                     ROOT / "reference_ablation/results" / variant if variant in PREVIOUS else
                     HERE / "results" / variant)
        prediction_files = sorted(directory.glob("*_predictions.npz"))
        if {p.name for p in prediction_files} != expected_names:
            raise RuntimeError(f"{variant}: incomplete/extra jobs: {len(prediction_files)}/29")
        for path in prediction_files:
            meta_path = path.with_name(path.name.replace("_predictions.npz", ".json"))
            meta = json.loads(meta_path.read_text())
            control = json.loads((control_dir / meta_path.name).read_text())
            if variant != CONTROL:
                if meta["variant"] != variant or meta["reference_update"] != (manifest["variants"] if variant in CANDIDATES else previous_definitions)[variant]:
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
    comparison.to_csv(HERE / "combined_comparison.csv", index=False)
    comparison[comparison.variant.isin(DISPLAY_VARIANTS)].to_csv(HERE / "comparison.csv", index=False)
    pd.DataFrame(paired).to_csv(HERE / "combined_paired_loso.csv", index=False)
    pd.DataFrame(paired).query("variant in @CANDIDATES").to_csv(HERE / "paired_loso.csv", index=False)
    verification = {
        "candidate_jobs": 145, "previous_candidate_jobs": 87, "control_jobs": 29, "officially_rescored_files": len(verified),
        "all_scores_match_official_code": True, "metric_atol": 1e-14,
        "all_original_artifacts_unchanged": True, "all_previous_ablation_artifacts_unchanged": True, "source_checkouts_unchanged": True,
        "data_hashes_match_control": True, "all_tree_selections_match_control": True,
        "all_test_window_orders_match_control": True, "combined_paired_loso_rows": len(paired), "new_paired_loso_rows": sum(r["variant"] in CANDIDATES for r in paired),
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
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    short = ['Control\nlog-Euclidean', 'Harmonic', 'Power-Euclidean\np=1/2',
             'Symmetric KL', 'Log-Cholesky', 'Bures-Wasserstein\nrecursive']
    colors = ['#667085', '#0077b6', '#2a9d8f', '#7b2cbf', '#c18b00', '#e76f51']
    fig, axes = plt.subplots(2,2,figsize=(13,8.5))
    for ax, model in zip(axes.flat, MODEL_LABELS):
        subset = comparison[comparison.model==model].set_index('variant').loc[list(DISPLAY_VARIANTS)]
        ax.bar(range(6),100*subset.loso_mean,color=colors,alpha=.85,
               yerr=100*subset.loso_std,capsize=4)
        ax.axhline(100/3,color='gray',linestyle='--',linewidth=1)
        ax.set(xticks=range(6),xticklabels=short,ylim=(0,85),ylabel='LOSO balanced accuracy (%)',
               title={'logreg':'One classifier','forest12':'Forest: 12 channels','forest20':'Forest: 20 channels'}[model])
        ax.tick_params(axis='x',labelsize=7)
    paired=pd.read_csv(HERE/'paired_loso.csv')
    ax=axes[1,1]
    for variant,color,marker in zip(CANDIDATES,colors[1:],['o','s','^','D','x']):
        g=paired[(paired.model=='forest20')&(paired.variant==variant)].sort_values('subject')
        ax.plot(range(8),100*g.delta,marker+'-',color=color,label=short[list(DISPLAY_VARIANTS).index(variant)].replace('\n',' '))
    ax.axhline(0,color='gray',linewidth=1)
    ax.set(xticks=range(8),xticklabels=[s.rsplit('/',1)[-1] for s in g.subject],
           ylabel='Change vs control (percentage points)',title='Forest 20: paired change per person')
    ax.legend(fontsize=7)
    fig.suptitle('Five literature-based SPD references — same splits, seed, trees and classifiers\nError bars: sample SD across 8 subjects (not confidence intervals)',fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE/'comparison.png',dpi=170)
    plt.close(fig)


def make_report(comparison, verification, manifest):
    new=comparison[comparison.variant.isin(DISPLAY_VARIANTS)]
    lines=[
      '# Ещё пять опорных матриц из литературы: PROTEUS', '',
      'Выполнены **145 новых оценок**: пять вариантов × тот же протокол из 29 задач. '
      'Добавлены прежние 116 оценок для совместного сравнения девяти способов построения опорной матрицы. '
      'Все сохранённые предсказания (261 файл) независимо проверены официальным кодом метрик.', '',
      'Это опубликованные методы оценивания SPD-матриц, применённые здесь как сессионные опорные матрицы '
      'в том же EEG-пайплайне. Это не воспроизведение пяти полных опубликованных систем распознавания ЭЭГ.', '',
      '## Результаты пяти новых вариантов', '',
      'Метрика — сбалансированная точность, средняя по сессиям. LOSO: среднее ± выборочное SD по восьми людям. '
      'Δ — парная разность с исходным лог-евклидовым накопительным вариантом, в процентных пунктах.', '']
    for model,label in MODEL_LABELS.items():
        lines += [f'### {label}', '',
          '| Опорная матрица | LOSO | Δ LOSO, п.п. | Лучше / равно / хуже по людям | Поздняя сессия | Test |',
          '|---|---:|---:|---:|---:|---:|']
        for r in new[new.model==model].itertuples():
            test='—' if pd.isna(r.test) else f'{r.test:.6f}'
            wins='—' if r.variant==CONTROL else f'{r.loso_wins} / {r.loso_ties} / {r.loso_losses}'
            lines.append(f'| {LABELS[r.variant]} | {r.loso_mean:.6f} ± {r.loso_std:.6f} | '
                         f'{100*r.delta_loso:+.2f} | {wins} | {r.later:.6f} | {test} |')
        lines.append('')
    lines += ['![Сравнение новых вариантов](comparison.png)', '', '## Наблюдения', '']
    for model,label in MODEL_LABELS.items():
        candidate=new[(new.model==model)&(new.variant!=CONTROL)].sort_values('loso_mean',ascending=False).iloc[0]
        overall=comparison[comparison.model==model].sort_values('loso_mean',ascending=False).iloc[0]
        lines.append(f'- {label}: лучший LOSO среди пяти новых вариантов — «{LABELS[candidate.variant]}», '
                     f'{candidate.loso_mean:.6f}, Δ {100*candidate.delta_loso:+.2f} п.п.; '
                     f'среди всех девяти вариантов максимум у «{LABELS[overall.variant]}», {overall.loso_mean:.6f}.')
    lines += ['', 'Сопоставление на публичном test для леса с 20 каналами:', '']
    for r in new[(new.model=='forest20')&(new.variant!=CONTROL)].itertuples():
        lines.append(f'- {LABELS[r.variant]}: {r.test:.6f}, Δ к контролю {100*r.delta_test:+.2f} п.п.')
    lines += ['', '## Все девять вариантов: лес с 20 каналами', '',
      '| Опорная матрица | LOSO | Поздняя сессия | Test |', '|---|---:|---:|---:|']
    for r in comparison[comparison.model=='forest20'].itertuples():
        lines.append(f'| {LABELS[r.variant]} | {r.loso_mean:.6f} ± {r.loso_std:.6f} | {r.later:.6f} | {r.test:.6f} |')
    lines += ['', '## Источники и точные формулы', '',
      'Обозначения: общий prior M0=exp(L0), где L0 — среднее логарифмов обучающих ковариаций; '
      'n=5+t; A=(5 M0+Σ R_i)/n; H=((5 M0⁻¹+Σ R_i⁻¹)/n)⁻¹. Начальная матрица и вес 5 одинаковы с предыдущим экспериментом.', '',
      '1. **Гармоническое среднее:** M=H. Это усреднение в пространстве обратных ковариаций. '
      'Формула H дана в [Nielsen (2024), Appendix A](https://pmc.ncbi.nlm.nih.gov/articles/PMC11675394/), '
      'с отсылкой к [Moakher & Batchelor (2006)](https://link.springer.com/chapter/10.1007/3-540-31272-2_17).', '',
      '2. **Степенное евклидово, p=1/2:** M=((5 M0^(1/2)+Σ R_i^(1/2))/n)^2. '
      '[Dryden, Pennec & Peyrat (2010), section 2.1](https://arxiv.org/abs/1009.3045). '
      'Это power-Euclidean mean, а не неявно определяемое риманово power mean; p=.5 зафиксировано до запусков.', '',
      '3. **Симметричная KL / Jeffreys:** M=A#H, где # — середина аффинно-инвариантной геодезической. '
      'Это центр для ковариаций нормальных распределений с одинаковым средним. '
      '[Nielsen (2024), Theorem A1, equation A5](https://pmc.ncbi.nlm.nih.gov/articles/PMC11675394/). '
      'Проверено также уравнение M H⁻¹ M=A. Это не формула центроида произвольных гистограмм.', '',
      '4. **Log-Cholesky:** для C_i=chol(R_i) строго нижние элементы C_i усредняются арифметически, '
      'а диагонали — через среднее логарифмов; затем M=Cbar Cbarᵀ. '
      '[Lin (2019), section 4.2, equation 4.3](https://arxiv.org/abs/1908.09326). '
      'Вес prior включён в тех же координатах. Порядок каналов сохранён; эта геометрия от него зависит.', '',
      '5. **Bures–Wasserstein, рекурсивное:** M_t=γ_BW(M_(t−1),R_t,1/(5+t)). '
      '[Bhatia, Jain & Lim (2019), section 4, equations 38–41](https://arxiv.org/abs/1712.01504). '
      'Геодезическая проверена независимой формулой оптимального переноса. '
      'Рекурсивный запуск — наша потоковая адаптация геодезической, не точный batch-барицентр всех окон.', '',
      'У первых четырёх вариантов вычисляется точное соответствующее взвешенное среднее/центр текущего префикса '
      'с массой prior 5. Все формулы применяются только к текущему и прошлым окнам сессии. '
      'Касательные признаки по-прежнему vec√2(log(M⁻¹/² R M⁻¹/²)); геометрия классификатора не заменялась.', '',
      '## Сохранённый протокол', '',
      '- Те же версии библиотек и подготовленные PROTEUS-файлы, проверенные SHA-256.',
      '- Те же 1521/379/755 окон, 6/2/2 человека, восемь LOSO-разбиений и единственная поздняя сессия.',
      '- C=.01, seed=0, 50 деревьев, m=12/20, исходные шесть полос, OAS и бутстреп-код автора.',
      '- Те же полосы и наборы каналов каждого дерева проверены сравнением с контрольными JSON.',
      '- Новое обновление используется и в обучении, и в предсказании; scaler/классификатор переобучаются с прежними настройками.',
      '- Текущее окно включается до классификации; состояние сбрасывается между сессиями.',
      '- Для public test обучение только на train; forest12 test не добавлялся, как и раньше.',
      '- Синтетическая часть с фиксированной единичной опорной точкой не изменялась.', '',
      '## Проверки и пределы вывода', '',
      f'- Официальными метриками проверены все предсказания ({verification["officially_rescored_files"]} файл): '
      '145 новых, 87 из прошлого сравнения и 29 исходных; все четыре метрики совпали при atol=1e-14.',
      '- Исходный код, данные и все предыдущие результаты остались неизменными; их хеши проверены.',
      '- Проверены скалярные ответы вручную, некоммутирующие взвешенные средние, уравнение KL-центра, '
      'формула BW-переноса, SPD, причинность, сброс сессий, пакетный/потоковый ответ и точное равенство контрольного адаптера.',
      '- Параметры пяти вариантов зафиксированы до получения новых оценок. Подбора по test не было.',
      '- Один seed, восемь dev-испытуемых, одна поздняя сессия и два public-test испытуемых. '
      'Результаты исследовательские: перебор нескольких методов и повторное использование открытого test '
      'не дают независимого подтверждения универсального превосходства. SD — разброс по людям, не доверительный интервал.', '',
      '## Файлы', '',
      '- `README.md`: команды запуска.',
      '- `experiment.json`: формулы, первоисточники, зафиксированные параметры и хеши.',
      '- `comparison.csv`: контроль + пять новых вариантов, 18 строк.',
      '- `combined_comparison.csv`: все девять вариантов, 27 строк.',
      '- `paired_loso.csv`: 120 новых парных разностей; `combined_paired_loso.csv`: 192 разности.',
      '- `fold_scores.csv`: 261 оценка; `results/<variant>/*_predictions.npz`: 145 новых файлов предсказаний.',
      '- `verification.json`, `verified_jobs.json`: независимая проверка метрик.', '',
    ]
    (HERE/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
