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
from representations import CONTROL, CANDIDATES, expanded_channels

VARIANTS = (CONTROL,) + CANDIDATES
LABELS = {
    CONTROL: "Исходная OAS-ковариация",
    "delay1_oas": "Ковариация с задержкой 1 отсчёт",
    "hsic_rbf": "HSIC, Gaussian RBF",
    "envelope_oas": "Ковариация огибающих",
}
MODEL_LABELS = {"logreg": "Один классификатор", "forest12": "Лес, 12 каналов", "forest20": "Лес, 20 каналов"}


def main():
    manifest = json.loads((HERE / "experiment.json").read_text())
    if original_snapshot() != manifest["original_artifact_sha256"]:
        raise RuntimeError("Original control artifacts changed during the comparison")
    if previous_snapshot() != manifest["previous_artifact_sha256"]:
        raise RuntimeError("Previous reference ablation artifacts changed")
    inventory = cache_inventory()
    if inventory["manifest_sha256"] != manifest["cache"]["manifest_sha256"]:
        raise RuntimeError("Frozen cache manifest changed")
    for tag, receipt in inventory["receipts"].items():
        d = 82 if tag.startswith("delay1_oas") else 41
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
        directory = control_dir if variant == CONTROL else HERE / "results" / variant
        prediction_files = sorted(directory.glob("*_predictions.npz"))
        if {p.name for p in prediction_files} != expected_names:
            raise RuntimeError(f"{variant}: incomplete/extra jobs: {len(prediction_files)}/29")
        for path in prediction_files:
            meta_path = path.with_name(path.name.replace("_predictions.npz", ".json"))
            meta = json.loads(meta_path.read_text())
            control = json.loads((control_dir / meta_path.name).read_text())
            if variant != CONTROL:
                if (meta["variant"] != variant or meta["representation_definition"] != manifest["variants"][variant]
                    or meta["reference_update"] != "original cumulative log-Euclidean, prior_weight=5"):
                    raise AssertionError("Variant configuration mismatch")
                cache_hashes = {tag:r["sha256"] for tag,r in inventory["receipts"].items() if tag.startswith(variant+"_")}
                if meta["matrix_cache_sha256"] != cache_hashes:
                    raise AssertionError("Training used another matrix cache")
                physical = {"logreg":41,"forest12":12,"forest20":20}[meta["model"]]
                d = physical * (2 if variant == "delay1_oas" else 1)
                if meta["tangent_dimension"] != d*(d+1)//2:
                    raise AssertionError("Wrong tangent-space dimension")
                if meta["model"] != "logreg":
                    expected = [(expanded_channels(t["channels"]).tolist() if variant == "delay1_oas" else t["channels"]) for t in control["trees"]]
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
    comparison.to_csv(HERE / "comparison.csv", index=False)
    pd.DataFrame(paired).to_csv(HERE / "paired_loso.csv", index=False)
    verification = {
        "candidate_jobs": 87, "control_jobs": 29, "officially_rescored_files": len(verified),
        "all_scores_match_official_code": True, "metric_atol": 1e-14,
        "all_original_artifacts_unchanged": True, "source_checkouts_unchanged": True,
        "all_previous_ablation_artifacts_unchanged": True, "all_18_matrix_cache_hashes_match": True,
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


def make_plots(frame, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    short = ["Original\nOAS", "Delay 1 sample\nOAS", "HSIC\nRBF", "Envelope\nOAS"]
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
    fig.suptitle("SPD representation comparison — original cumulative log-Euclidean reference\nError bars: sample SD across 8 subjects (not confidence intervals)", fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "comparison.png", dpi=170)
    plt.close(fig)


def make_report(comparison, verification, manifest):
    lines = [
        "# Замена оконной матрицы R(X) в исходном пайплайне PROTEUS", "",
        "Выполнены 87 новых оценок: ковариация с задержкой, HSIC Gram и ковариация амплитудных огибающих. "
        "Сравнение с 29 исходными оценками; накопительное лог-евклидово обновление опорной матрицы сохранено. "
        "Все 116 файлов предсказаний заново оценены официальным кодом соревнования.", "",
        "## Результаты", "",
        "Метрика — balanced accuracy, средняя по сессиям (`session_mean`), как в воспроизведённой таблице 2. "
        "LOSO: среднее ± выборочное SD по восьми людям. Δ — изменение относительно исходного представления "
        "для того же классификатора, в процентных пунктах. SD отражает разброс людей, а не доверительный интервал.", "",
    ]
    for model, label in MODEL_LABELS.items():
        lines += [f"### {label}", "",
                  "| Представление R(X) | LOSO | Δ LOSO, п.п. | Лучше / равно / хуже по людям | Поздняя сессия | Test |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in comparison[comparison.model == model].itertuples():
            test = "—" if pd.isna(r.test) else f"{r.test:.6f}"
            wins = "—" if r.variant == CONTROL else f"{r.loso_wins} / {r.loso_ties} / {r.loso_losses}"
            lines.append(f"| {LABELS[r.variant]} | {r.loso_mean:.6f} ± {r.loso_std:.6f} | {100*r.delta_loso:+.2f} | {wins} | {r.later:.6f} | {test} |")
        lines.append("")
    lines += ["![Сравнение и парные изменения](comparison.png)", "", "## Наблюдения", ""]
    for model, label in MODEL_LABELS.items():
        best = comparison[comparison.model == model].sort_values("loso_mean", ascending=False).iloc[0]
        lines.append(f"- {label}: максимальное среднее LOSO у «{LABELS[best.variant]}», "
                     f"{best.loso_mean:.6f}; Δ к исходному представлению {100*best.delta_loso:+.2f} п.п.")
    forest = comparison[comparison.model == "forest20"].set_index("variant")
    lines += ["", "Для леса с 20 физическими электродами на публичном test:", ""]
    for variant in CANDIDATES:
        r = forest.loc[variant]
        lines.append(f"- {LABELS[variant]}: {r.test:.6f}, Δ {100*r.delta_test:+.2f} п.п.")
    lines += ["", "## Что именно изменено", "",
        "В авторском коде исходное R(X) уже вычисляется оценкой **OAS**, а не буквально необработанным XXᵀ. "
        "Чтобы сравнивать преобразование сигнала, OAS сохранён для задержек и огибающих. "
        "Следовательно, эти два варианта — регуляризованные ковариации OAS, а не точное Cov+εI из обсуждения. "
        "Для HSIC применяется указанная ниже добавка εI.", "",
        "Перед каждым представлением — исходный Butterworth bandpass порядка 4 и sosfiltfilt, "
        "для тех же шести полос: 4–40, 4–8, 8–13, 13–20, 20–30 и 30–40 Гц.", "",
        "1. **Исходная OAS:** Covariances('oas') на отфильтрованных 41×480 отсчётах.",
        "2. **Задержка:** z_i(t)=[x_i(t),x_i(t−1)], t=1,…,479, затем OAS. "
        "При частоте 120 Гц задержка равна 8.333 мс. Это один фиксированный лаг; поиск лага и глубины не проводился. "
        "Обе копии каждого электрода выбираются вместе, физические наборы деревьев совпадают с исходными.",
        "3. **HSIC:** для каждого канала K_i(t,s)=exp(−(x_i(t)−x_i(s))²/(2σ_i²)), "
        "σ_i — медиана |x_i(t)−x_i(s)| по всем t<s внутри этого окна; при σ_i≤1e−12 используется 1. "
        "G_i=HK_iH, H=I−11ᵀ/T; S_ij=tr(G_iG_j)/T²; R=S+εI, "
        "ε=max(1e−12,1e−6·mean(diag(S))). Используются **все 480 отсчётов**, "
        "без прореживания, random Fourier features или Nyström. Упаковка симметричных ядер сохраняет "
        "их точное скалярное произведение Фробениуса. SPD гарантирована Gram-конструкцией и ε>0.",
        "4. **Огибающая:** a_i(t)=abs(hilbert(x_i))(t), затем OAS по всем 480 отсчётам. "
        "Считается ковариация огибающих; нормировка до корреляции не добавляется.", "",
        "### Размерность", "",
        "| R(X) | Полная матрица | Матрица дерева 12 / 20 электродов | Признаки logreg / forest12 / forest20 |",
        "|---|---:|---:|---:|",
        "| Исходная / HSIC / огибающая | 41×41 | 12×12 / 20×20 | 861 / 78 / 210 |",
        "| С задержкой | 82×82 | 24×24 / 40×40 | 3403 / 300 / 820 |", "",
        "## Опорная матрица и обучение", "",
        "Для каждого разбиения и полосы prior переоценивается на **обучающих** новых матрицах: "
        "L₀=mean(log R_train). Исходный SessionRecentering наследуется без замены: "
        "L_t=(5L₀+Σᵢ₌₁ᵗ log R_i)/(5+t), M_t=exp(L_t). Текущее окно входит в обновление до классификации. "
        "В начале каждой сессии состояние сбрасывается к prior. Численное значение prior меняется вместе "
        "с представлением; формула и вес 5 остаются исходными. StandardScaler и классификатор переобучаются.", "",
        "- Те же данные PROTEUS v1.0.0: train/val/test = 1521/379/755 окон, 6/2/2 человека, 41 канал, 4 секунды.",
        "- Те же восемь LOSO на train+val, поздняя сессия и public test; тот же порядок окон и правила сброса.",
        "- Test обучается только на исходных шести train-испытуемых; forest12 test отсутствует, как в исходной таблице.",
        "- Те же C=0.01, seed=0, 50 деревьев, 12/20 физических электродов, полосы и исходный код бутстрепа.",
        "- Ни одна из ранее проверенных альтернативных опорных матриц здесь не используется.",
        "- Синтетическая часть статьи этим запуском не изменялась; воспроизводится тот же EEG-протокол таблицы 2.", "",
        "## Кэш и проверки", "",
        "Чистое преобразование отдельного окна кэшируется до обучения. Для 2655 окон рассчитаны "
        "18 наборов матриц (3 представления × 6 полос). Преобразование не использует метки, другие окна "
        "или обучаемые статистики; bandwidth HSIC берётся только из своего окна. Наличие test в таком "
        "кэше не переносит его информацию в обучающий prior или StandardScaler. Поиск окна — по точному "
        "хешу исходного float32 массива; неизвестное окно вызывает ошибку, а не подменяется похожим.", "",
        f"- Официальным кодом заново оценены {verification['officially_rescored_files']} файлов: "
        "session_mean, cell_mean, pooled balanced accuracy и accuracy совпали с сохранёнными при atol=1e−14.",
        "- Проверены хеши всех 18 файлов матриц, положительность минимального собственного значения каждого окна, "
        "размерности признаков, порядок окон, физические электроды и группировка временных копий.",
        "- Все исходные 29 результатов, оба предыдущих сравнения опорных матриц, данные и авторские репозитории неизменны.",
        "- 65 тестов: исходные проверки, прошлые методы опорной матрицы и новые аналитические проверки HSIC, "
        "задержек, огибающих, SPD, пакетного/потокового ответа, причинности по порядку окон, "
        "равенства живого и кэшированного обучения, возобновления прерванного кэша.", "",
        "## Пределы вывода", "",
        "Параметры lag=1, медианный bandwidth и ridge зафиксированы до результатов; "
        "оптимизация по public test не проводилась. Это сравнение конкретных конфигураций на восьми "
        "dev-испытуемых и двух public-test испытуемых с одним seed; оно не проверяет все задержки "
        "или все способы моделирования нелинейных связей. Изменение размерности задержанного варианта "
        "также меняет число признаков при прежнем C. Отдельной настройки регуляризации не проводилось.", "",
        "HSIC измеряет силу зависимости без знака обычной корреляции. Медианный bandwidth по каждому "
        "окну/каналу в значительной мере убирает информацию об абсолютной мощности; порядок отсчётов "
        "внутри окна не моделируется. Огибающая также изменяет доступную фазовую информацию. "
        "Прирост задержанного варианта получен одновременно с увеличением числа признаков; этот эксперимент не отделяет вклад задержек от роста ёмкости модели.", "",
        "Кэш рассчитан офлайн для сравнения точности. Время fit в JSON использует готовые матрицы и "
        "не является полной задержкой реального приложения. SOS-фильтрация и Hilbert используют "
        "всё доступное 4-секундное окно, как оконный протокол, без заявления о причинности внутри окна. "
        "Открытый test уже использовался ранее и не заменяет закрытую финальную оценку соревнования.", "",
        "## Файлы", "",
        "- `README.md`: команды повторного запуска.",
        "- `representations.py`: три определения R(X), адаптеры исходных моделей и группировка каналов.",
        "- `matrix_cache.py`, `prepare_cache.py`: точный возобновляемый кэш; `cache_inventory.json`: его хеши и SPD-проверки.",
        "- `experiment.json`: исходные хеши, окружение, параметры и формулы до получения результатов.",
        "- `comparison.csv`, `fold_scores.csv`, `paired_loso.csv`: 12 итоговых строк, 116 отдельных оценок, 72 парных изменения.",
        "- `results/<variant>/*_predictions.npz`: новые предсказания каждого окна.",
        "- `verification.json`, `verified_jobs.json`: независимая проверка.", "",
        "## Источники", "",
        "- [Carrara & Papadopoulo, 2023 — Augmented Covariance Matrix](https://arxiv.org/abs/2302.04508): "
        "мотивирует временное расширение. Здесь фиксирован один лаг, без воспроизведения их поиска параметров.",
        "- [Gretton et al., 2005 — HSIC](https://www.cs.cmu.edu/~arthurg/papers/GreBouSmoSch05.pdf): "
        "основа меры зависимости. Используемая здесь SPD-матрица между каналами — предложенная в переписке "
        "Gram-конструкция, а не опубликованный алгоритм-победитель соревнования.",
        "- [Corsi et al., FUCONE](https://arxiv.org/html/2111.03122v2): рассматривает amplitude envelope correlation. "
        "Здесь проверяется предложенная ковариация огибающих; полный FUCONE не воспроизводится.",
        f"- [Код автора](https://github.com/DorinDaniil/Brain-Decoding), commit `{manifest['source_commit']}`.",
        f"- [Официальный код соревнования](https://github.com/neural-interfaces26/2026-competition), commit `{manifest['competition_commit']}`.", "",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
