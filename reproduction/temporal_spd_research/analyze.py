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
from run import original_snapshot, previous_snapshot, cache_inventory, sha256, write_json
from temporal_features import CONTROL, CANDIDATES, MULTIPLICITY, expanded_channels
from select_model import validate_selection

VARIANTS = (CONTROL,"delay1_oas") + CANDIDATES
LABELS = {CONTROL:"Исходная OAS", "delay1_oas":"Задержка 1 (прежний лидер)",
 "lag4_oas":"Задержка 4", "stationary_oas":"Стационарная тёплицева", "pyramid2_oas":"Две половины окна",
 "quadrature_oas":"Комплексная / квадратурная", "path_integral_oas":"Интегральная ковариация пути", "signature2_oas":"OAS + сигнатурный Gram"}
MODEL_LABELS = {"logreg": "Один классификатор", "forest12": "Лес, 12 каналов", "forest20": "Лес, 20 каналов"}


def main():
    selection = validate_selection()
    manifest = json.loads((HERE / "experiment.json").read_text())
    if original_snapshot() != manifest["original_artifact_sha256"]:
        raise RuntimeError("Original control artifacts changed during the comparison")
    if previous_snapshot() != manifest["previous_artifact_sha256"]:
        raise RuntimeError("Previous reference ablation artifacts changed")
    inventory = cache_inventory()
    if inventory["manifest_sha256"] != manifest["cache"]["manifest_sha256"]:
        raise RuntimeError("Frozen cache manifest changed")
    for tag, receipt in inventory["receipts"].items():
        d = 41*MULTIPLICITY[tag.rsplit("_band",1)[0]]
        if receipt["shape"] != [2655,d,d] or receipt["min_eigenvalue"] <= 0:
            raise AssertionError("Cache dimension/SPD check failed")
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
        directory = control_dir if variant == CONTROL else (ROOT/"representation_ablation/results/delay1_oas" if variant=="delay1_oas" else HERE/"results"/variant)
        prediction_files = sorted(directory.glob("*_predictions.npz"))
        if {p.name for p in prediction_files} != expected_names:
            raise RuntimeError(f"{variant}: incomplete/extra jobs: {len(prediction_files)}/29")
        for path in prediction_files:
            meta_path = path.with_name(path.name.replace("_predictions.npz", ".json"))
            meta = json.loads(meta_path.read_text())
            control = json.loads((control_dir / meta_path.name).read_text())
            if variant in CANDIDATES:
                if (meta["variant"] != variant or meta["representation_definition"] != manifest["variants"][variant]
                    or meta["reference_update"] != "original cumulative log-Euclidean, prior_weight=5"):
                    raise AssertionError("Variant configuration mismatch")
                cache_hashes = {tag:r["sha256"] for tag,r in inventory["receipts"].items() if tag.startswith(variant+"_")}
                if meta["matrix_cache_sha256"] != cache_hashes:
                    raise AssertionError("Training used another matrix cache")
                physical = {"logreg":41,"forest12":12,"forest20":20}[meta["model"]]
                d = physical * MULTIPLICITY[variant]
                if meta["tangent_dimension"] != d*(d+1)//2:
                    raise AssertionError("Wrong tangent-space dimension")
                if meta["model"] != "logreg":
                    expected = [(expanded_channels(t["channels"]).tolist() if MULTIPLICITY[variant]==2 else t["channels"]) for t in control["trees"]]
                    if meta["matrix_channels"] != expected:
                        raise AssertionError("Physical electrode copies were not grouped")
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
            records.append({k: v for k, v in dict(meta, variant=variant).items() if k not in ("trees", "matrix_channels", "matrix_cache_sha256")})
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
    best_rows = frame[(frame.variant=="delay1_oas") & (frame.protocol=="loso")]
    for index,row in comparison.iterrows():
        candidate = frame[(frame.variant==row.variant)&(frame.model==row.model)&(frame.protocol=="loso")].set_index("fold").session_mean.sort_index()
        oldbest=best_rows[best_rows.model==row.model].set_index("fold").session_mean.sort_index()
        delta=candidate-oldbest
        comparison.loc[index,"delta_loso_vs_delay1"]=delta.mean()
        comparison.loc[index,"wins_vs_delay1"]=int((delta>1e-12).sum())
    ranked=sorted(comparison[comparison.model=="forest20"].to_dict("records"),key=lambda r:(-r["loso_mean"],r["variant"]))
    if ranked[0]["variant"]!=selection["winner"]:raise AssertionError("Declared selection does not match primary metric")
    comparison.to_csv(HERE / "comparison.csv", index=False)
    pd.DataFrame(paired).to_csv(HERE / "paired_loso.csv", index=False)
    verification = {
        "candidate_jobs": 174, "control_jobs": 58, "officially_rescored_files": len(verified),
        "all_scores_match_official_code": True, "metric_atol": 1e-14,
        "all_original_artifacts_unchanged": True, "source_checkouts_unchanged": True,
        "all_previous_ablation_artifacts_unchanged": True, "all_36_matrix_cache_hashes_match": True, "selection_frozen_before_test": True, "selected_variant":selection["winner"],
        "all_cached_windows_strictly_positive_definite": True, "all_delay_copies_grouped": True,
        "original_logeuclidean_updater_inherited_unchanged": True,
        "data_hashes_match_control": True, "all_tree_selections_match_control": True,
        "all_test_window_orders_match_control": True, "paired_loso_rows": len(paired),
        "source_commit": manifest["source_commit"], "competition_commit": manifest["competition_commit"],
    }
    write_json(HERE / "cache_inventory.json", inventory)
    write_json(HERE / "verification.json", verification)
    write_json(HERE / "verified_jobs.json", verified)
    make_plots(frame, comparison)
    make_report(comparison, verification, manifest)
    print(comparison.to_string(index=False))
    print(json.dumps(verification, indent=2))
    print(HERE / "REPORT.md")


def make_plots(frame,comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels=["Original", "Delay 1", "Delay 4", "Stationary", "Two halves", "Quadrature", "Path integral", "Signature"]
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    colors=["#667085","#0077b6","#00a896","#f4a261","#9b5de5","#e76f51","#457b9d","#bc6c25"]
    for ax,model in zip(axes.flat,MODEL_LABELS):
        r=comparison[comparison.model==model].set_index("variant").loc[list(VARIANTS)]
        ax.bar(range(8),100*r.loso_mean,yerr=100*r.loso_std,capsize=3,color=colors)
        ax.axhline(100/3,color="gray",ls="--",lw=1)
        ax.set(xticks=range(8),xticklabels=labels,ylim=(0,85),ylabel="LOSO balanced accuracy (%)",title=MODEL_LABELS[model].replace("Один классификатор","One classifier").replace("Лес, ","Forest: ").replace(" каналов"," channels"))
        ax.tick_params(axis="x",rotation=35,labelsize=9)
    ax=axes[1,1]
    r=comparison[comparison.model=="forest20"].set_index("variant").loc[list(VARIANTS)]
    ax.bar(range(8),100*r.test,color=colors)
    ax.axhline(100*r.loc["delay1_oas","test"],color="#0077b6",ls="--",label="Previous best")
    ax.set(xticks=range(8),xticklabels=labels,ylim=(0,75),ylabel="Public-test balanced accuracy (%)",title="Forest 20: fixed candidates, no test selection")
    ax.tick_params(axis="x",rotation=35,labelsize=9);ax.legend(fontsize=9)
    fig.suptitle("Changing only R(X): six literature-driven representations\nOriginal cumulative log-Euclidean reference; error bars are sample SD across 8 people",fontsize=13)
    fig.tight_layout();fig.savefig(HERE/"comparison.png",dpi=160);plt.close(fig)


def make_report(comparison,verification,manifest):
    selection=validate_selection();winner=selection["winner"]
    lines=["# Исследование функции R(X): временная структура и непрерывность", "",
      f"Наибольшее сырое среднее LOSO леса с 20 физическими электродами до численной фильтрации: **{LABELS[winner]}**. "
      "Выбор зафиксирован до новых test-оценок. Это сырое ранжирование первого этапа. У блочного и квадратурного представлений обнаружены теоретически нулевые касательные координаты, усиливаемые исходным scaler; рекомендацию см. в FINAL_REPORT.md и протоколе followup_mixed/PROTOCOL.md.", "",
      "Изменялась только оконная функция R(X). Выполнены 174 новых оценки, сопоставлены с 58 сохранёнными "
      "оценками исходного OAS и предыдущего лидера delay1. Исходный бейзлайн и все предыдущие результаты сохранены.", "",
      "## Результаты", "",
      "Balanced accuracy, средняя по сессиям. LOSO: среднее ± выборочное SD по восьми людям. "
      "Δ — изменение относительно прежнего лидера с задержкой 1, в процентных пунктах. SD не является доверительным интервалом.", ""]
    for model,label in MODEL_LABELS.items():
      lines += [f"### {label}","", "| R(X) | LOSO | Δ к delay1, п.п. | Лучше delay1, людей | Поздняя сессия | Public test |", "|---|---:|---:|---:|---:|---:|"]
      for r in comparison[comparison.model==model].itertuples():
        test="—" if pd.isna(r.test) else f"{r.test:.6f}"
        lines.append(f"| {LABELS[r.variant]} | {r.loso_mean:.6f} ± {r.loso_std:.6f} | {100*r.delta_loso_vs_delay1:+.2f} | {int(r.wins_vs_delay1)}/8 | {r.later:.6f} | {test} |")
      lines.append("")
    lines += ["![Сравнение](comparison.png)","", "## Что проверено в литературе", "",
      "Сохранён обзор девяти первичных работ из анализа временных рядов, динамических систем, компьютерного зрения, "
      "спектрального анализа и сигнатур траекторий: [literature/survey.md](literature/survey.md). "
      "Каждая заметка отделяет утверждения статьи от нашей адаптации. Ниже дано точное содержание эксперимента; "
      "полные алгоритмы TICC, Cov3DJ, сигнатурного классификатора и спектральной вейвлет-регрессии не воспроизводятся.", "",
      "### 1. Задержка 4 отсчёта", "",
      "Z_i(t)=[x_i(t),x_i(t−4)], R=OAS(Z), 476 совмещённых отсчётов, задержка 33.333 мс. "
      "Те же 82 координаты, что у delay1. Основание — augmented covariance / delay embedding "
      "[Carrara & Papadopoulo, 2023](https://arxiv.org/abs/2302.04508). Лаг зафиксирован, поиск гиперпараметров статьи не копируется.", "",
      "### 2. Стационарная блочно-тёплицева ковариация", "",
      "Из каждого канала вычитается среднее окна. Строятся два вектора длины T+1: [x,0] и [0,x]. "
      "Их Gram имеет вид [[C₀,C₁],[C₁ᵀ,C₀]] в блочном порядке, после чего применяется OAS. "
      "Это PSD по построению, а усадка сохраняет тёплицеву структуру и даёт SPD на данных. "
      "Мотивировано стационарными автоковариациями у [Wu & Xiao](https://www.stat.uchicago.edu/~wbwu/papers/cov-reviewJune15-2011-2.pdf) "
      "и структурированными зависимостями у [Hallac et al., TICC](https://arxiv.org/abs/1706.03161). "
      "Сегментация, кластеризация и graphical lasso в наш бейзлайн не добавляются.", "",
      "### 3. Ковариации двух половин окна", "",
      "R=diag(OAS(X_first_half), OAS(X_second_half)), координаты переупорядочены по физическому электроду. "
      "Это один уровень временной иерархии, сохраняющий порядок участков. "
      "Основание — [Hussein et al., 2013](https://www.ijcai.org/Proceedings/13/Papers/363.pdf). "
      "Блочная упаковка в одну SPD-матрицу и исходный классификатор — наша адаптация.", "",
      "### 4. Комплексная / квадратурная ковариация", "",
      "Q=Im(Hilbert(X)); S=OAS([X,Q]); R=(S+J S Jᵀ)/2, где J поворачивает каждую пару (x_i,q_i) на 90°. "
      "Сохраняется вещественная форма эрмитовой ковариации, включая квадратурные межканальные связи. "
      "Оба слагаемых SPD, их среднее SPD. Мотивировано многомерными спектральными матрицами "
      "[von Sachs, 2020](https://doi.org/10.1146/annurev-statistics-031219-041138). "
      "Это интегральное оконное представление; частотная вейвлет-регрессия из литературы не реализована.", "",
      "### 5. Интегральная ковариация непрерывного пути", "",
      "Между отсчётами выбирается линейная интерполяция. Для каждого интервала с концами a,b берутся "
      "узлы z±=(a+b)/2 ± (b−a)/(2√3). Эмпирические первые и вторые моменты этих 2(T−1) узлов "
      "точно равны интегральным моментам кусочно-линейного пути; R=OAS(z). Матрица остаётся 41×41. "
      "Интерполяция явным образом учитывает соседство отсчётов. Это предложенная здесь конструкция, "
      "мотивированная подходом к последовательности как пути; она не приписывается авторам сигнатурных методов. "
      "У OAS меняется число входных узлов, поэтому меняется и усадка — это часть проверяемого R(X).", "",
      "### 6. Исходная ковариация плюс сигнатурный Gram", "",
      "R=C_OAS(X)+s² ΦΦᵀ/64. s — один RMS всего окна; Φ содержит по четыре координаты "
      "(S_x,S_tx,S_xt,S_xx) сигнатуры порядка 2 для 16 непересекающихся отрезков по 30 отсчётов каждого канала. "
      "Время каждого отрезка 0..1, амплитуда делится на s. Интегралы вычисляются точно на линейных интервалах "
      "по формуле Чена; данные не прореживаются. Фиксированные отрезки сцепляются, однако Gram суммирует по их координатам и теряет абсолютный порядок отрезков. "
      "Прибавка — PSD Gram, исходная ковариация сохраняет мощность и обеспечивает SPD. "
      "Механизм опирается на [Király & Oberhauser, 2019](https://jmlr.org/papers/v20/16-314.html) "
      "и [Dyer et al., 2022](https://proceedings.mlr.press/v151/dyer22a.html); такая смесь между каналами — наша адаптация. "
      "Универсальность полного сигнатурного ядра не переносится автоматически на глубину 2 и эту упаковку. Отдельная проверка выявила инвариантность этого Gram к полному обращению времени: координаты меняются ортогонально, а Gram остаётся тем же.", "",
      "## Что означает непрерывность", "",
      "Дискретные отсчёты не задают единственную непрерывную кривую. Линейная интерполяция — явное модельное "
      "предположение. Интегральная ковариация усредняет значения кривой; сигнатура использует упорядоченные "
      "интегралы приращений. Ковариация пути инвариантна обращению времени; временная пирамида сохраняет "
      "порядок половин; задержки сохраняют направленные поперечные лаги. Эти представления сохраняют разные свойства.", "",
      "## Что осталось исходным", "",
      "Авторские репозитории и данные не изменены. Для всех функций используется исходная полосовая фильтрация "
      "Butterworth/SOS, исходные шесть полос, mean(log R_train) как prior, накопительный SessionRecentering с весом 5, "
      "StandardScaler, C=.01, seed=0, 50 деревьев и те же 12/20 физических электродов. Для матриц 82×82 "
      "обе координаты электрода выбираются вместе. Test обучается только на исходных шести train-испытуемых.", "",
      "| Семейство | Полная матрица | Признаки logreg / forest12 / forest20 |", "|---|---:|---:|",
      "| Исходная, интегральная, сигнатурная | 41×41 | 861 / 78 / 210 |",
      "| Delay1, lag4, stationary, pyramid, quadrature | 82×82 | 3403 / 300 / 820 |", "",
      "Число независимых ненулевых признаков блочных и комплексных конструкций может быть меньше формальной размерности. "
      "C не настраивался; это сравнение замен R(X) при фиксированном бейзлайне, а не лучших классификаторов для каждого представления.", "",
      "## Проверки и границы вывода", "",
      f"- Пересчитаны официальным кодом все {verification['officially_rescored_files']} файла предсказаний, четыре метрики совпали при atol=1e−14.",
      "- Проверены 36 хешей кэша и положительность всех 95 580 оконных матриц, размерности, физические наборы деревьев и порядок окон.",
      "- 32 новых теста проверяют аналитические формулы, SPD, пакетный/потоковый ответ, сброс сессии, причинность по окнам, живое/кэшированное обучение и восстановление кэша.",
      "- Снимки хешей всех трёх предыдущих серий и исходного контроля совпали.",
      "- Протокол коммитнут до расчётов; выбор LOSO-лидера коммитнут до новых test-оценок. Подбора по ним не было.",
      "- Выбор среди шести функций на восьми dev-людях всё равно является исследовательским поиском: оценка выбранного максимума LOSO оптимистична. Открытый test ранее использовался; закрытая независимая проверка не заменена.",
      "- Один seed, фиксированные lag4/16 patches/два временных участка. Это не перебор всех SPD-функций и не доказательство отсутствия других лучших представлений.",
      "- Время fit исключает построение кэша; нельзя трактовать его как полную задержку приложения. Hilbert и sosfiltfilt используют доступное целое окно.", "",
      "## Воспроизведение и артефакты", "",
      "[README.md](README.md) — команды. [PROTOCOL.md](PROTOCOL.md) — доэкспериментальный протокол. "
      "[comparison.csv](comparison.csv) — 24 строки итогов; [fold_scores.csv](fold_scores.csv) — 232 оценки; "
      "[paired_loso.csv](paired_loso.csv) — парные разности с исходной OAS. "
      "[selection.json](selection.json), [verification.json](verification.json), [cache_inventory.json](cache_inventory.json) — проверяемая доказательная база.", ""]
    (HERE/"REPORT.md").write_text("\n".join(lines))


if __name__=="__main__":main()
