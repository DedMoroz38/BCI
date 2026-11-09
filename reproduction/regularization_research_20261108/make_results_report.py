"""Human report and selected train/validation plots; no training or tuning."""
import json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/proteus-regularization-mpl')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;OUT=HERE/'to_human';ROOT=HERE/'downloads/gpu-v2/regularization-results'
OLD=HERE.parent/'metric_research_20261108'

def main():
    table=pd.read_csv(OUT/'REGULARIZATION_COMPARISON.csv');mean=table[table.seeds=='mean2'].set_index('config')
    old=pd.read_csv(OLD/'to_human/GPU_ALL10_COMPARISON.csv').set_index('variant')
    baseline=old.loc['shallow16_class_mean2__mix_T4_balance1'];oldcnn=old.loc['shallow16_class_mean2__cnn']
    folds=pd.read_csv(OUT/'REGULARIZATION_FOLD_SCORES.csv');people=pd.read_csv(OLD/'to_human/GPU_PERSON_SCORES.csv')
    b=people[people.variant=='shallow16_class_mean2__mix_T4_balance1'].set_index('tag').cell_mean
    v=folds[(folds.config=='swad0.1')&(folds.seeds=='mean2')&folds.tag.str.startswith('loso_')].set_index('tag').mix_cell
    delta=v-b
    pd.DataFrame(dict(previous=b,swad=v,delta=delta)).to_csv(OUT/'SWAD_PERSON_DELTAS.csv')
    verify=json.loads((OUT/'REGULARIZATION_VERIFICATION.json').read_text())
    lines=['# Регуляризация: результаты Kaggle Version 2 — 9 ноября 2026','',
        'Все 35 конфигураций завершились: 770 обучений, два seed, 10 LOSO-фолдов и один фолд поздних сессий. Ошибок моделей и прерванных по бюджету задач нет. Обучение заняло 1 час 22 минуты. Отдельные контроли не переобучались по запросу пользователя; используются сохранённые результаты прошлой волны.', '',
        'Метрика — cell balanced accuracy по объединённым 2655 OOF-предсказаниям 10 людей. Каждый человек исключён из обучения своей LOSO-модели. Поздние сессии двух людей считаются отдельно. Все строки ниже используют среднее вероятностей двух seeds, без выбора удачного seed.', '',
        '| Метод | CNN отдельно | Смесь с Lag4 | Поздние сессии: смесь | Δ смеси к предыдущему варианту |','|---|---:|---:|---:|---:|',
        f'| Предыдущий вариант без новой регуляризации | {oldcnn.official_oof_cell*100:.2f}% | {baseline.official_oof_cell*100:.2f}% | {baseline.later_cell*100:.2f}% | — |']
    chosen=[('swad0.1','SWAD, ratio 0.1'),('ls05_noise03_drop3','LS 0.05 + шум 3% + маскирование 3 координат'),
        ('shift12','Временной сдвиг до ±100 мс'),('ema0.99','EMA 0.99'),('global_gain0.1','Общая амплитуда ±10%'),
        ('ls0.1','Label smoothing 0.1'),('ls0.2','Label smoothing 0.2'),('mixstyle0.3','MixStyle 0.3'),
        ('drop3','Маскирование 3 координат'),('window5','Среднее последних 5 эпох')]
    for key,label in chosen:
        r=mean.loc[key];lines.append(f'| {label} | {r.cnn_oof_cell*100:.2f}% | {r.mix_oof_cell*100:.2f}% | {r.mix_late_cell*100:.2f}% | {(r.mix_oof_cell-baseline.official_oof_cell)*100:+.2f} п.п. |')
    swad=mean.loc['swad0.1']
    lines += ['',
        f'Лучший по этому CV — SWAD: {swad.mix_oof_cell*100:.2f}% в смеси, прирост {(swad.mix_oof_cell-baseline.official_oof_cell)*100:.2f} п.п. к прошлой смеси и {(swad.mix_oof_cell-old.loc["lag4","official_oof_cell"])*100:.2f} п.п. к Lag4. Выигрыш у {int((delta>0).sum())}/10 людей, проигрыш у {int((delta<0).sum())}/10. Поздние сессии: {baseline.later_cell*100:.2f}% → {swad.mix_late_cell*100:.2f}%.', '',
        f'CNN со SWAD отдельно: {oldcnn.official_oof_cell*100:.2f}% → {swad.cnn_oof_cell*100:.2f}% (+{(swad.cnn_oof_cell-oldcnn.official_oof_cell)*100:.2f} п.п.), всё ещё ниже Lag4 {old.loc["lag4","official_oof_cell"]*100:.2f}%. Рецепт смеси фиксирован: 70% Lag4 + 30% среднего softmax(CNN logits / 4), затем причинная балансировка alpha1/prior10. Вес и температура заново не подбирались.', '',
        '## Что получилось', '',
        'Усреднение SWAD и EMA — наиболее перспективные направления этой серии. Обычное среднее нескольких последних эпох проиграло прошлой смеси: принцип выбора диапазона и весов имеет значение. По этой серии нельзя приписать эффект только усреднению: его варианты включают пересчёт BatchNorm на чистых обучающих данных; отдельный BN-контроль был исключён пользователем.', '',
        'Простые аугментации дали небольшие и неоднородные эффекты. Временной сдвиг и слабое общее масштабирование оказались полезнее маскирования координат, длинных временных масок и смешивания окон. Это вывод по нашей сетке, а не универсальная рекомендация для EEG.', '',
        f'Label smoothing 0.2 снизил OOF NLL CNN с {oldcnn.log_loss:.3f} до {mean.loc["ls0.2","cnn_nll"]:.3f}, при этом BA CNN {mean.loc["ls0.2","cnn_oof_cell"]*100:.2f}% и смесь {mean.loc["ls0.2","mix_oof_cell"]*100:.2f}%. Улучшение вероятностного loss не превратилось в сопоставимый рост конкурсной метрики.', '',
        '## Переобучение осталось', '',
        'Пример P01, seed0: к остановке SWAD-кривая имеет train BA около 96,6%, val BA 55,6%; label smoothing 0.2 — около 99,8% и 51,5%. Это метрики конца внутреннего обучения, а не выбранного чекпоинта. Некоторые методы снизили NLL; большой разрыв обобщения между людьми сохраняется.', '',
        'Выбор эпох остаётся по внутреннему NLL, как запросил пользователь. SWAD дополнительно определяет диапазон усреднения по внутреннему NLL. Предобученные модели не использовались.', '',
        'Прирост победителя всего 0,39 п.п. после сравнения 35 вариантов на переиспользованной публичной CV. Он исследовательский; скрытый тест не измерялся. Результаты отдельных seeds заметно различаются, поэтому вывод делается по среднему двух seeds, а не по лучшему seed.', '',
        'Следующая гипотеза, если продолжать: SWAD + временные сдвиги и SWAD + label smoothing. Такие сочетания в этой серии не проверялись. Новых запусков при разборе результатов не выполнялось.', '',
        '## Артефакты', '',
        '[Все 35 вариантов и seeds](REGULARIZATION_COMPARISON.csv) · [Изменения по людям](SWAD_PERSON_DELTAS.csv) · [Кривые](REGULARIZATION_CURVES.png) · [Пересчёт результатов](REGULARIZATION_VERIFICATION.json)', '',
        f'Все 770 сохранённых оценок пересчитаны из предсказаний; метаданные и хеши предсказаний/логитов сверены. Скачано и проверено {verify["learning_histories"]} историй выбранных методов и {verify["downloaded_best_model_hashes"]} моделей SWAD. Полные состояния оптимизаторов не скачивались.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    choices=[('previous','Previous CNN'),('ls0.2','Label smoothing 0.2'),('ema0.99','EMA 0.99'),('swad0.1','SWAD 0.1')]
    fig,axes=plt.subplots(2,4,figsize=(15,6.5),sharex='col')
    for col,(name,label) in enumerate(choices):
        if name=='previous':
            folder=OLD/'downloads/gpu-v2/metric-results/models/loso_P01/shallow16_class/seed0/selection'
        else:folder=ROOT/'models/loso_P01'/name/'seed0/selection'
        h=pd.read_csv(folder/'curves.csv');r=json.loads((folder/'complete.json').read_text())
        for row,key in enumerate(('ce','cell_ba')):
            ax=axes[row,col]
            for prefix,color in [('train','#2473b6'),('val','#d64b35')]:
                values=h.dropna(subset=[prefix+'_'+key]);ax.plot(values.epoch,values[prefix+'_'+key],color=color,marker='o',ms=3,label=prefix)
            ax.axvline(r['best_epoch'],color='gray',ls='--',label='NLL-selected epoch');ax.grid(alpha=.2)
            if row==0:ax.set_title(label);ax.set_ylabel('Cross entropy')
            else:ax.set_ylabel('Cell balanced accuracy');ax.set_xlabel('Epoch');ax.set_ylim(.3,1.02)
    axes[0,0].legend(fontsize=8);fig.suptitle('P01, seed0: clean training / inner held-out people (not outer test)')
    fig.tight_layout();fig.savefig(OUT/'REGULARIZATION_CURVES.png',dpi=150);plt.close(fig)
    print('REPORT',OUT/'RESULTS.md',flush=True)

if __name__=='__main__':main()
