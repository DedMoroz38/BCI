"""Build a Russian report solely from completed, verified measurements."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
eeg = pd.read_csv(ROOT / 'results/eeg/comparison.csv').set_index('model')
manifest = json.loads((ROOT / 'results/eeg/manifest.json').read_text())
verification = json.loads((ROOT / 'results/eeg/verification.json').read_text())
assert verification['verified_prediction_files'] == 29
syn = pd.read_csv(ROOT / 'results/synthetic/table1.csv')
syn_verify = json.loads((ROOT / 'results/synthetic/verification.json').read_text())

names = {'logreg': 'Один касательный классификатор', 'forest12': 'Лес, m=12', 'forest20': 'Лес, m=20'}
lines = [
 '# Воспроизведение эксперимента «Риманов лес»', '',
 'Эксперимент на ЭЭГ выполнен неизменённым опубликованным кодом автора. '
 'Синтетическая часть реконструирована по разделу 5: её исходники, seed и часть деталей не опубликованы. '
 'Таким образом, работа не является побитовым воспроизведением всей статьи.', '',
 '## Таблица 2: измеренные результаты на PROTEUS', '',
 'Метрика — сбалансированная точность, усреднённая по сессиям. '
 'Для LOSO указано среднее ± выборочное стандартное отклонение по восьми испытуемым.', '',
 '| Модель | LOSO: статья | LOSO: повтор | Поздняя сессия: статья / повтор | Test: статья / повтор |',
 '|---|---:|---:|---:|---:|',
]
for name, r in eeg.iterrows():
 test = '—' if pd.isna(r.get('test')) else f"{r.paper_test:.3f} / {r.test:.6f}"
 lines.append(f'| {names[name]} | {r.paper_loso_mean:.3f} ± {r.paper_loso_std:.3f} | '
              f'{r.loso_mean:.6f} ± {r.loso_std:.6f} | {r.paper_later:.3f} / {r.later:.6f} | {test} |')
matches = []
for name, r in eeg.iterrows():
 for metric in ['loso_mean', 'loso_std', 'later', 'test']:
  if pd.notna(r.get('paper_' + metric)):
   matches.append(dict(model=name, metric=metric, measured=float(r[metric]),
                       paper=float(r['paper_'+metric]),
                       matches_at_3_decimals=f'{r[metric]:.3f}' == f'{r["paper_"+metric]:.3f}'))
(ROOT / 'results/eeg/paper_agreement.json').write_text(json.dumps(matches, indent=2))
if all(m['matches_at_3_decimals'] for m in matches):
 lines += ['', '**Все независимо пересчитанные значения моделей таблицы 2 совпали со статьёй '
           'при её точности представления — три знака после запятой.**']
else:
 lines += ['', '**Часть значений таблицы 2 не совпала при округлении до трёх знаков. '
           'Точные расхождения сохранены в comparison.csv и paper_agreement.json.**']
lines += ['', 'Расхождения указаны в `results/eeg/comparison.csv`; '
          'параметры не подбирались для совпадения с опубликованными числами.', '',
          '## Что именно запускалось', '',
          '- PROTEUS, NEMAR nm000290 v1.0.0; 41 канал, 4 секунды, 120 Гц.',
          '- Исходная подготовка `bci.data`: notch 50/60 Гц, полоса 0.1–75 Гц до ресэмплинга, '
          'окно начинается через 1 секунду после подсказки.',
          '- Разбиение по людям с random_state=33; train/val/test: 1521/379/755 окон, 6/2/2 человека.',
          '- Baseline: полоса 4–40 Гц, C=0.01, prior_weight=5.',
          '- Леса: 50 деревьев, m=12/20, шесть исходных полос, C=0.01, prior_weight=5, seed=0 '
          '(значение по умолчанию в опубликованных ноутбуках).',
          '- 24 LOSO-задачи, 3 оценки поздней сессии, 2 оценки публичного test. '
          'Для test обучение только на train, как в исходных ноутбуках; m=12 на test не оценивался.',
          '- В потоковом предсказании каждое окно подаётся отдельно; состояние сбрасывается между сессиями. '
          'Используется исходное накопительное среднее без добавления забывания.', '',
          'Опубликованные результаты EEGNet/CTNet/REVE в статье использованы только как внешние ориентиры. '
          'Эти нейросети здесь не переобучались.', '',
          '## Независимая проверка', '',
          f'- Все {verification["verified_prediction_files"]} файлов предсказаний повторно оценены '
          'кодом `benchmark_utils/metrics.py` официального соревнования; '
          'оценки совпали с сохранёнными при абсолютном допуске 1e-14.',
          '- Проверены непересечение людей между исходными частями и между train/test каждого LOSO, '
          'число окон, форма, наличие классов и порядок окон.',
          '- Дополнительно проверено совпадение пакетных и потоковых предсказаний исходных baseline/forest '
          'на искусственных окнах с несколькими потоками.',
          '- Хеши подготовленных данных, версии библиотек и списки испытуемых: `results/eeg/manifest.json`.', '',
          '## Таблица 1: реконструкция синтетического эксперимента', '',
          'Здесь повторены 16 каналов, точные логарифмы ковариационных подматриц, '
          '60 обучающих и 2000 тестовых окон, sigma=0.1, 50 деревьев, 10 задач и пять C от 0.01 до 100. '
          'Лучшее C выбирается по test, как явно сделано в статье; это оптимистичная оценка.', '',
          '| Метод | m | Статья: среднее ± SD | Реконструкция: среднее ± SD |',
          '|---|---:|---:|---:|']
paper = {('forest',4):(.777,.037),('forest',8):(.761,.021),('forest',12):(.728,.024),
         ('structured',4):(.794,.021),('structured',8):(.761,.024),('structured',12):(.730,.028),
         ('full',16):(.702,.027)}
method_names = {'forest':'Лес','structured':'Структурированная регрессия','full':'Все каналы, логистическая регрессия'}
for r in syn.itertuples():
 a,b=paper[(r.method,r.m)]
 lines.append(f'| {method_names[r.method]} | {r.m} | {a:.3f} ± {b:.3f} | {r.mean:.5f} ± {r.std:.5f} |')
lines += ['', 'Выбранные до запуска неопределённые детали, включая seed, число деревьев для рисунка 1а '
          'и способ усреднения дисперсии, перечислены в `results/synthetic/manifest.json`. '
          'Эти решения не выдаются за настройки автора.', '',
          '## Рисунок 1: реконструкция', '',
          '![Реконструкция рисунка 1](results/synthetic/figure1.png)', '',
          f'Максимальное абсолютное отклонение средней корреляции от теории: '
          f'{syn_verify["correlation_max_abs_error_of_task_mean"]:.5f}. '
          f'Максимальное абсолютное отклонение оценки относительной дисперсии от теории: '
          f'{syn_verify["variance_max_abs_error"]:.5f} (включая K=1; оценки получены ресэмплингом конечного пула).', '',
          'При C=0.001 относительная ошибка эквивалентности леса и структурированной регрессии составила '
          f'{100*syn_verify["equivalence_at_C_0.001"]["0.02"]:.2f}% при sigma=0.02 и '
          f'{100*syn_verify["equivalence_at_C_0.001"]["0.15"]:.2f}% при sigma=0.15. '
          'В статье приведены примерно 1.3% и 10%; точное совпадение не получено.', '',
          '## Ограничения точного воспроизведения', '',
          'Большинство версий в requirements автора не закреплено; исходное окружение неизвестно. '
          'Текущие установленные версии закреплены в `environment.lock.txt`. '
          'Неизвестны исходные seed и часть процедур синтетики. '
          'На основании этих запусков нельзя утверждать, что вся статья воспроизведена один-в-один.', '',
          '## Источники и файлы', '',
          '- Статья: `../Riemannian_Forest (1).pdf`, раздел 5, страницы 14–17.',
          '- Код автора: https://github.com/DorinDaniil/Brain-Decoding, commit '+manifest['source_commit']+'.',
          '- Код соревнования: https://github.com/neural-interfaces26/2026-competition, commit '
          +verification['competition_commit']+'.',
          '- Данные: https://doi.org/10.82901/nemar.nm000290, лицензия CC-BY-4.0; '
          'Dreyer et al., PROTEUS BCI Bordeaux, v1.0.0.',
          '- Команды повторного запуска: `README.md`.',
          '- Все оценки по отдельным задачам: `results/eeg/fold_scores.csv`.',
          '- Сырые данные реконструкции: `results/synthetic/*.csv`.', '',
          '### Численные библиотеки', '']
lines += [f'- {name}=={version}' for name,version in manifest['package_versions'].items()]
diagnostic_path = ROOT / 'results/synthetic/tolerance_diagnostic.csv'
if diagnostic_path.exists():
 diagnostic = pd.read_csv(diagnostic_path)
 tight = diagnostic[diagnostic.tol == 1e-10].groupby('sigma').relative_error.mean()
 lines += ['', '### Дополнительная диагностика численного решателя', '',
           'В отдельном запуске при C=0.001 уменьшение допуска tol с 1e-4 до 1e-10 '
           f'дало относительные ошибки {100*tight.loc[.02]:.2f}% и {100*tight.loc[.15]:.2f}%. '
           'Это подтверждает чувствительность сравнения при сильной регуляризации к точности оптимизации. '
           'Основные результаты реконструкции выше оставлены без изменений; '
           'неизвестно, какой допуск использовал автор. '
           'Скрипт: scripts/diagnose_tolerance.py; данные: results/synthetic/tolerance_diagnostic.csv.']
lines += ['', '### Разбиение по испытуемым', '']
for name,s in manifest['splits'].items():
 lines.append(f'- {name}: {s["windows"]} окон, {s["subjects"]} человек, IDs: {", ".join(s["subject_ids"])}.')
(ROOT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
print(ROOT / 'REPORT.md')
