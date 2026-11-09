# Пять дополнительных опорных матриц из литературы

Рабочая папка: `/Users/den1shh/Documents/diploma`. Используется существующее
`reproduction/.venv`, прежние данные и неизменённый код автора/соревнования.

## Методы и источники

| Ключ | Опорная матрица | Источник |
|---|---|---|
| harmonic_cumulative | Гармоническое среднее / усреднение обратных ковариаций | Moakher & Batchelor (2006); Nielsen (2024), Appendix A |
| poweuclid_half | Степенное евклидово, p=1/2 | Dryden, Pennec & Peyrat (2010), section 2.1 |
| kullback_sym | Центр симметричной KL / Jeffreys для центрированных Gaussian | Nielsen (2024), Theorem A1 / A5 |
| logchol_cumulative | Log-Cholesky mean | Lin (2019), section 4.2 / equation 4.3 |
| wasserstein_recursive | Индуктивное обновление по Bures–Wasserstein геодезической | Bhatia, Jain & Lim (2019), section 4 / equations 38–41 |

Ссылки и точные формулы зафиксированы в `literature_references.py` и `experiment.json`.
Это применение опубликованных SPD-оценок как опорных матриц в нашем EEG-пайплайне,
а не репликация пяти полных EEG-систем. Для BW используется потоковая индуктивная
оценка, а не точный batch-барицентр.

Начальный prior у всех тот же, лог-евклидов по обучению, вес 5. Изменяется только
сессионная опорная матрица. Фильтрация, OAS, классификаторы, C, seed, полосы,
случайные подмножества и бутстреп-код совпадают с исходным экспериментом.

## Запуск

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/reproduction/.cache/matplotlib"

# 43 проверки: прежние + новые формулы, причинность, контрольный адаптер.
reproduction/.venv/bin/python -m pytest -q \
  reproduction/literature_ablation/test_literature_references.py \
  reproduction/reference_ablation/test_references.py reproduction/scripts

# Зафиксировать настройки и проверить хеши/окружение до вычислений.
reproduction/.venv/bin/python -u reproduction/literature_ablation/run.py --initialize

# 145 новых оценок (5 × 29), до четырёх независимых локальных процессов.
# Завершённые задачи пропускаются при повторном запуске.
reproduction/.venv/bin/python -u reproduction/literature_ablation/launch.py --workers 4

# Официальная проверка всех 261 предсказаний, сравнение, графики и отчёт.
reproduction/.venv/bin/python reproduction/literature_ablation/analyze.py
```

Отдельный вариант/модель:

```bash
reproduction/.venv/bin/python -u reproduction/literature_ablation/run.py \
  --variant logchol_cumulative --model forest20
```

Доступны `--protocol cv` и `--protocol test`. Как в исходном протоколе, test
вычисляется для logreg и forest20, обученных только на train. Для forest12
дополнительного test нет. Параметры по test не подбираются.

## Результаты

- `experiment.json`: конфигурация, формулы, первоисточники и хеши до первого запуска.
- `results/<variant>/`: 29 JSON + 29 NPZ на новый вариант, всего 145 новых предсказаний.
- `logs/`: выполнение групп и проверки.
- `comparison.csv`: контроль + пять новых вариантов, 18 строк для трёх моделей.
- `combined_comparison.csv`: все девять вариантов, 27 строк.
- `paired_loso.csv`: 120 новых парных сравнений по людям.
- `combined_paired_loso.csv`: 192 сравнения с учётом предыдущих методов.
- `fold_scores.csv`: 261 оценка, включая 116 прежних.
- `verification.json`, `verified_jobs.json`: независимая проверка официальными метриками.
- `REPORT.md`, `comparison.png`: отчёт и график.

Формулы/данные/пакеты защищены хешами от смешения разных конфигураций при продолжении.
Не запускайте две копии одной группы одновременно в одну папку результатов.
