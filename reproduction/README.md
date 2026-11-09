# Воспроизведение «Риманова леса»

Рабочая папка: `/Users/den1shh/Documents/diploma`.

- `Brain-Decoding/`: неизменённый код автора, commit `8d577af42619e5911cc5da8743c8a3e9b629014b`.
- `2026-competition/`: неизменённый код соревнования, commit `2b53aec91c111f8b5423b6674f096c48cc1a99af`.
- `.venv/`: отдельный Python 3.12.12 с зависимостями из requirements автора.
- `environment.lock.txt`: фактически установленные версии.
- `data/`: PROTEUS v1.0.0 и подготовленные окна; данные CC-BY-4.0.
- `results/eeg/`: оценки, предсказания, параметры и хеши данных.
- `results/synthetic/`: реконструкция синтетической части с явно указанными допущениями.
- `logs/`: журналы подготовки и вычислений.

## Повторный запуск

В терминале из рабочей папки:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/reproduction/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/reproduction/.cache"
export PYTHONPATH="$PWD/reproduction/Brain-Decoding/track2_bci"

# Подготовка данных исходным кодом автора, требуется интернет (~1,5 ГБ).
reproduction/.venv/bin/python -u -m bci.data --data-dir reproduction/data

# Все 29 оценок таблицы 2: 24 LOSO, 3 поздние сессии, 2 тестовые.
# Сохраняет результат каждой задачи и пропускает уже завершённые задачи.
reproduction/.venv/bin/python -u reproduction/scripts/reproduce_eeg.py

# Независимая проверка всех оценок официальным кодом метрик соревнования.
reproduction/.venv/bin/python reproduction/scripts/verify_results.py

# Реконструкция рисунка 1 и таблицы 1; это не исходные скрипты автора.
reproduction/.venv/bin/python -u reproduction/scripts/reproduce_synthetic.py

# Проверки защиты протокола и матричных преобразований.
reproduction/.venv/bin/python -m pytest -q reproduction/scripts
```

Для нового независимого запуска EEG используйте `--output reproduction/results/eeg-new-run`.
Тест оценивается только для baseline и выбранного леса m=20, обучение — на train,
как в исходных ноутбуках. Test не используется для подбора параметров EEG.

Синтетическая таблица 1 повторяет описанный в статье выбор лучшего C по тесту;
такая оценка оптимистична. Скрипт не подбирает seed или иные параметры ради совпадения чисел.

## Предел точности воспроизведения

Исходное окружение автора не опубликовано: большая часть requirements не закреплена.
Установленные сейчас версии зафиксированы в `environment.lock.txt`; при расхождениях
нельзя приписывать их самому методу до проверки исходных версий.

Для синтетики не опубликованы исходный код, seed и часть деталей выборки/усреднения.
Известные параметры повторяются, остальные объявлены в `results/synthetic/manifest.json`.
Результаты нейросетевых baseline в статье — внешние опубликованные ориентиры; здесь
они не обучаются заново. Главная проверка таблицы 2 — один касательный классификатор
и два римановых леса.

Дополнительная диагностика чувствительности синтетического сравнения при сильной
регуляризации к допуску решателя (основной запуск и EEG не меняет):

```bash
reproduction/.venv/bin/python -u reproduction/scripts/diagnose_tolerance.py
```
