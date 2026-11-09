"""Readable summary and learning-curve artifacts from verified saved results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE/'to_human'
ROOT = HERE/'downloads/gpu-v2/metric-results'


def main():
    table = pd.read_csv(OUT/'GPU_ALL10_COMPARISON.csv').set_index('variant')
    diagnostic = pd.read_csv(OUT/'GPU_LEARNING_DIAGNOSTICS.csv')
    verify = json.loads((OUT/'GPU_VERIFICATION.json').read_text())
    assert verify['cpu_model_replay'] == 'all55_passed'
    models = [('lag4','Lag4'), ('lag4_balance1','Lag4 + балансировка'),
        ('shallow40_class_s0__cnn','Только CNN40, seed0'),
        ('shallow16_class_mean2__cnn','Только CNN16, class loss, среднее 2 seeds'),
        ('shallow16_subject_mean2__cnn','Только CNN16, subject loss, среднее 2 seeds'),
        ('shallow40_class_s0__mix_T4_balance1','Lag4 + CNN40 + балансировка'),
        ('shallow16_class_mean2__mix_T4_balance1','Lag4 + CNN16 class, 2 seeds + балансировка'),
        ('shallow16_subject_mean2__mix_T4_balance1','Lag4 + CNN16 subject, 2 seeds + балансировка')]
    lines = ['# Kaggle Wave1: проверенные результаты Version 2', '',
        'Завершено 55 обучений: 11 внешних фолдов × 3 конфигурации с seed0, плюс 11 × 2 компактные конфигурации с seed1. Основной пакет занял 520 секунд (8 минут 40 секунд), без подготовки среды. Пропущенных фолдов нет.', '',
        'Проверены SHA-256 кода, данных, предсказаний, весов, выравнивания и логитов. Все 55 метрик пересчитаны независимо; все 55 моделей повторно выполнили инференс на CPU с восстановленным причинным выравниванием. Классы совпали точно, максимальное отличие вероятности 1,26e-6 при допуске 1e-4. Проверены внутренние разбиения по людям и число эпох финального переобучения.', '',
        '## Сравнение на одинаковых фолдах', '',
        'Основная метрика — cell balanced accuracy по объединённым 2655 OOF-предсказаниям для всех 10 людей. В каждой LOSO-модели соответствующий человек отсутствует в обучении. Поздний фолд проверяет последние сессии двух людей. Среднее по людям приведено отдельно. С прежним LOSO на 8 людях сравнивать числа напрямую нельзя.', '',
        '| Метод | OOF cell BA | Среднее по людям | Поздние сессии |', '|---|---:|---:|---:|']
    for key, label in models:
        row = table.loc[key]
        lines.append(f'| {label} | {row.official_oof_cell*100:.2f}% | {row.mean_person_cell*100:.2f}% | {row.later_cell*100:.2f}% |')
    person = pd.read_csv(OUT/'GPU_PERSON_SCORES.csv')
    base = person[person.variant == 'lag4'].set_index('tag').cell_mean
    chosen = 'shallow16_class_mean2__mix_T4_balance1'
    delta = person[person.variant == chosen].set_index('tag').cell_mean-base
    lines += ['',
        f'Практический кандидат: 70% вероятностей Lag4 + 30% среднего softmax(logits / 4) двух CNN16 с class loss; затем причинная балансировка alpha=1, uniform prior=10. OOF cell BA {table.loc[chosen,"official_oof_cell"]*100:.2f}%, выигрыш к Lag4 {(table.loc[chosen,"official_oof_cell"]-table.loc["lag4","official_oof_cell"])*100:.2f} п.п.; улучшение у {int((delta>0).sum())}/10 людей. Против Lag4 с балансировкой выигрыш {(table.loc[chosen,"official_oof_cell"]-table.loc["lag4_balance1","official_oof_cell"])*100:.2f} п.п.', '',
        'Рецепт смеси перенесён из предыдущего локального исследования без нового перебора весов/температур. Две CNN объединяются средним вероятностей, а не выбором лучшего seed. Максимальная строка отдельного seed — CNN16 subject seed1 в смеси (60,50%); это не основной результат, потому что выбор удачного seed по внешнему CV добавляет оптимизм. Выбор между архитектурами также остаётся исследовательским.', '',
        'Отдельные CNN уступают Lag4. Сравнение только seed0 не поддерживает уменьшение сети как улучшение качества: смеси CNN40 / CNN16 class / CNN16 subject дают 60,09% / 59,69% / 59,56%. Усреднение двух seed улучшило отдельные компактные CNN в этом CV, но сравнение 1 seed большой сети против 2 seeds маленькой не изолирует эффект размера.', '',
        '## Переобучение и критерий остановки', '',
        'В 51/55 внутренних кривых к остановке train CE ниже первого значения, а val CE выше. В 50 LOSO-обучениях минимум выбранного val NLL обычно на 1–3 эпохе; максимум наблюдавшейся val BA расположен позже в 49/50 случаях. Это расхождение критериев, а не доказательство того, что поздняя эпоха лучше на внешнем человеке.', '',
        '| Конфигурация, seed0 | Медиана выбранной эпохи | Средний train CE в конце | Средний val CE в конце | Train / val cell BA в конце |', '|---|---:|---:|---:|---:|']
    for config in ('shallow40_class','shallow16_class','shallow16_subject'):
        rows = diagnostic[(diagnostic.config == config)&(diagnostic.seed == 0)]
        lines.append(f'| {config} | {rows.best_epoch.median():.0f} | {rows.train_ce_last.mean():.3f} | {rows.val_ce_last.mean():.3f} | {rows.train_ba_last.mean()*100:.1f}% / {rows.val_ba_last.mean()*100:.1f}% |')
    lines += ['',
        'Пример фиксированного фолда P01, CNN40: выбранная по NLL эпоха 1 имеет val BA 44,34%; на эпохе 7 val BA 55,68%, одновременно val CE растёт с 1,107 до 1,616, train CE падает с 0,806 до 0,057. Модель может улучшать классы, одновременно становясь чрезмерно уверенной в ошибках. Этим объясняется возможный конфликт loss и конкурсной метрики, но не устанавливается причина всех ошибок.', '',
        'Следующая проверяемая гипотеза: выбирать эпоху по внутренней cell BA и отдельно смягчать вероятности для ансамбля, сохраняя вложенное разбиение по людям. Новый запуск для этой гипотезы не выполнялся. Результат на скрытом соревновании не измерялся; переиспользованный публичный CV не даёт независимого подтверждения.', '',
        '[Кривые P01 для всех трёх архитектур](GPU_LEARNING_CURVES.png). Столбцы — архитектуры, сверху CE, снизу cell BA. Вертикальная линия показывает выбранную по NLL эпоху; метрики конца обучения в таблице не являются метриками выбранного чекпоинта.', '',
        '[Все варианты и seeds](GPU_ALL10_COMPARISON.csv) · [Проверки](GPU_VERIFICATION.json) · [CPU replay](GPU_CPU_REPLAY.json)']
    (OUT/'GPU_REPORT.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':10})
    fig, axes = plt.subplots(2,3,figsize=(13,6.3),sharex='col')
    for col, config in enumerate(('shallow40_class','shallow16_class','shallow16_subject')):
        parent = ROOT/'models/loso_P01'/config/'seed0/selection'
        curves = pd.read_csv(parent/'curves.csv')
        best = json.loads((parent/'complete.json').read_text())['best_epoch']
        for row, key, ylabel in ((0,'ce','Cross entropy'),(1,'cell_ba','Cell balanced accuracy')):
            ax = axes[row,col]
            for prefix, color in (('train','#2473b6'),('val','#d64b35')):
                values = curves.dropna(subset=[prefix+'_'+key])
                ax.plot(values.epoch,values[prefix+'_'+key],marker='o',ms=3,label=prefix,color=color)
            ax.axvline(best,color='gray',ls='--',label='NLL-selected epoch')
            ax.set_ylabel(ylabel);ax.grid(alpha=.2)
            if row == 0:ax.set_title(config)
            else:ax.set_ylim(.3,1.02);ax.set_xlabel('Epoch')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('P01 outer fold, seed0: inner held-out people (not outer test)')
    fig.tight_layout();fig.savefig(OUT/'GPU_LEARNING_CURVES.png',dpi=150);plt.close(fig)
    print('REPORT',OUT/'GPU_REPORT.md',flush=True)


if __name__ == '__main__':main()
