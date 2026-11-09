# Готовый сабмишен: Riemann-Forest-Lag4

Загружать в Codabench нужно **riemann_forest_lag4.zip** целиком.

Track02: https://www.codabench.org/competitions/17982/
Официальная инструкция: https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/participate.md

1. Войти в Codabench и присоединиться к Track02.
2. Открыть My Submissions и выбрать активную фазу.
3. Загрузить riemann_forest_lag4.zip.
4. Дождаться статуса Finished; при ошибке warm-up посмотреть первый ingestion error.

## Что находится в ZIP

Все три файла лежат непосредственно в корне:

- submission.py — весь код инференса, Solver.load_model, predict и reset_state.
- weights.npz — обученные численные параметры без pickle или зависимостей от классов проекта.
- model_info.json — описание модели и происхождения весов.

Дополнительные Python-модули, исходные данные и исследовательский кэш в архив не включены.
Веса — финальный экземпляр уже проверенной конфигурации:50 моделей,20 физических
электродов в каждой,6 полос частот,лаг 4 отсчёта (33.333 мс),OAS,накопительное
лог-евклидово среднее с prior_weight5,C=.01,seed0.

Обучение:1521 окно исходного train,6 испытуемых. Validation и public-test метки
при обучении не использовались. Это тот же протокол, для которого измерены
приведённые ниже результаты; веса не переобучались на всех публичных людях.

## Проверено

- Все 755 test-предсказаний в 4 независимых потоках совпали с исследовательским прогоном.
- Проверка выполнялась после извлечения ZIP в папку только для чтения.
- Использован официальный benchmark_utils.streaming.predict_streams с deepcopy/reset_state.
- Импорты bci,pyriemann,temporal_features и других исследовательских модулей были запрещены при проверке.
- Новое окно, отсутствующее в исследовательском кэше, обрабатывается нормально.
- Локальный Benchopt успешно выполнил официальный objective на 48 реальных по размерности окнах.
- 109 тестов прошли, включая совпадение вероятностей с исходной реализацией,
  сброс состояния, независимость копий, порядок каналов и прежние численные тесты.

Полный public test:

| Метрика | Значение |
|---|---:|
| Cell-mean balanced accuracy — метрика текущего лидерборда |55.716363%|
| Session-mean balanced accuracy — наши предыдущие таблицы |55.540729%|
| Pooled balanced accuracy |55.585973%|
| Accuracy |55.629139%|

Это локальная оценка public test, а не результат загрузки на Codabench или
результат на скрытой sealed-выборке. Весь проход 755 окон занял около 22 секунд
на текущем CPU; это не гарантия времени на сервере.

Инференс использует numpy,scipy,scikit-learn и предоставляемый соревнованием
benchmark_utils. Нет зависимости от pyriemann,данных на диске или сохранённого
кэша окон. Модель работает на CPU и принимает torch-тензоры, включая перенос
входа с GPU на CPU. Метаданные проверяют 120 Гц,41 канал и 3 класса; перестановка
тех же именованных каналов поддерживается. Стандартный Simulated(8 каналов,
100 Гц) не соответствует этим обученным весам, поэтому проверка сделана на
реальных размерах PROTEUS.

## Файлы проверки

- verification.json — SHA256 ZIP,точное совпадение 755 ответов и метрики.
- build.json — версии пакетов,источник численных функций и время обучения.
- logs/validation.log — полная проверка извлечённого read-only ZIP.
- logs/benchopt.log — запуск официального Benchopt-контракта.
- logs/final-tests.log —109 прошедших тестов.

Benchopt сообщил, что sandbox не разрешает sysctl для определения названия CPU.
Это предупреждение о сведениях оборудования; solver завершился со статусом done.
Официальный objective.py скопирован без изменений в official_check; исходный
репозиторий соревнования не менялся. Linux/Docker worker непосредственно не запускался.

## Повторная проверка ZIP

Из /Users/den1shh/Documents/diploma:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/Users/den1shh/Documents/diploma/reproduction/.cache/matplotlib reproduction/.venv/bin/python reproduction/submission_lag4/validate_zip.py
```

Не включайте test_submission.py,build_weights.py,validate_zip.py,official_check
или logs в загружаемый архив. Готовый ZIP уже содержит ровно три нужных файла.
