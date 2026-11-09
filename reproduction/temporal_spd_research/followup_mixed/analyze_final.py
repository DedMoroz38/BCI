"""Verify the stable follow-up and combine all 261 same-protocol predictions."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;P=HERE.parent;ROOT=P.parent
sys.path.insert(0,str(ROOT/'2026-competition'))
from benchmark_utils.metrics import balanced_accuracy,group_scores
from run import sha256,original_snapshot,previous_snapshot,cache_inventory
from mixed_features import DEFINITIONS,expanded_channels
from mixed_selection import validate_selection

LABELS={'original_oas':'Исходная OAS','delay1_oas':'Задержка 1','lag4_oas':'Задержка 4','stationary_oas':'Стационарная тёплицева','pyramid2_oas':'Две половины: сырая блочная*','quadrature_oas':'Квадратурная*','path_integral_oas':'Интегральная ковариация пути','signature2_oas':'OAS + сигнатурный Gram','pyramid2_mixed':'Две половины: сумма/разность'}
MODELS={'logreg':'Один классификатор','forest12':'Лес,12 электродов','forest20':'Лес,20 электродов'}

def main():
 selection=validate_selection();manifest=json.loads((HERE/'experiment.json').read_text())
 primary=json.loads((P/'verification.json').read_text())
 for test_file in (P/'results').rglob('*_test_*.json'):
  assert test_file.stat().st_mtime>=pd.Timestamp(selection['created_utc']).timestamp()
 assert primary['officially_rescored_files']==232 and primary['all_scores_match_official_code']
 assert original_snapshot()==manifest['original_artifact_sha256']
 assert previous_snapshot()==manifest['previous_artifact_sha256']
 for name,digest in manifest['code_sha256'].items():assert sha256(HERE/name)==digest,name
 inventory=cache_inventory();assert inventory['manifest_sha256']==manifest['cache']['manifest_sha256']
 cache_hash={k:v['sha256'] for k,v in inventory['receipts'].items()}
 original=ROOT/'results/eeg';directory=HERE/'results/pyramid2_mixed'
 files=sorted(directory.glob('*_predictions.npz'));assert len(files)==29
 assert {f.name for f in files}=={f.name for f in original.glob('*_predictions.npz')}
 records=[];verified=[]
 for path in files:
  mpath=path.with_name(path.name.replace('_predictions.npz','.json'))
  meta=json.loads(mpath.read_text());control=json.loads((original/mpath.name).read_text())
  assert meta['representation_definition']==DEFINITIONS['pyramid2_mixed']
  assert meta['reference_update']=='original cumulative log-Euclidean, prior_weight=5'
  assert meta['matrix_cache_sha256']==cache_hash
  assert meta['near_null_scaler_coordinates']==0 and meta['minimum_scaler_variance']>1e-20
  for key in ['train_subjects','eval_subjects','train_windows','eval_windows','model','protocol','fold']:assert meta[key]==control[key]
  d={'logreg':82,'forest12':24,'forest20':40}[meta['model']]
  assert meta['tangent_dimension']==d*(d+1)//2
  if meta['model']!='logreg':
   assert meta['trees']==control['trees']
   assert meta['matrix_channels']==[expanded_channels(t['channels']).tolist() for t in control['trees']]
  if meta['protocol'] in ['loso','test']:assert not(set(meta['train_subjects'])&set(meta['eval_subjects']))
  if meta['protocol']=='test':assert mpath.stat().st_mtime>=pd.Timestamp(selection['created_utc']).timestamp()
  with np.load(path) as f,np.load(original/path.name) as b:
   for key in ['y','subject','session','context','run','onset']:np.testing.assert_array_equal(f[key],b[key])
   y,pred=f['y'],f['pred'];assert y.shape==pred.shape and set(np.unique(pred))<={0,1,2}
   stream=np.char.add(np.char.add(f['subject'],'/'),f['session']);cells=np.stack([stream,f['context']],axis=1)
   metrics=dict(session_mean=float(group_scores(balanced_accuracy,y,pred,stream).mean()),cell_mean=float(group_scores(balanced_accuracy,y,pred,cells).mean()),pooled=balanced_accuracy(y,pred),accuracy=float(np.mean(y==pred)))
   for k,v in metrics.items():assert np.isclose(v,meta[k],rtol=0,atol=1e-14),(path,k)
  assert np.isclose(meta['session_mean_delta_vs_control'],metrics['session_mean']-control['session_mean'],atol=1e-14,rtol=0)
  records.append({k:v for k,v in meta.items() if k not in ['trees','matrix_channels','matrix_cache_sha256']})
  verified.append({'job':path.stem,'official_scores_match':True})
 (HERE/'verified_jobs.json').write_text(json.dumps(verified,indent=2))
 frame=pd.concat([pd.read_csv(P/'fold_scores.csv'),pd.DataFrame(records)],ignore_index=True)
 assert len(frame)==261;frame.to_csv(P/'final_fold_scores.csv',index=False)
 rows=[]
 for variant in LABELS:
  for model in MODELS:
   r=frame[(frame.variant==variant)&(frame.model==model)]
   loso=r[r.protocol=='loso'].set_index('fold').session_mean.sort_index();assert len(loso)==8
   base=frame[(frame.variant=='original_oas')&(frame.model==model)&(frame.protocol=='loso')].set_index('fold').session_mean.sort_index()
   prev=frame[(frame.variant=='delay1_oas')&(frame.model==model)&(frame.protocol=='loso')].set_index('fold').session_mean.sort_index()
   assert loso.index.equals(prev.index)
   later=r[r.protocol=='later'].session_mean;test=r[r.protocol=='test'].session_mean
   assert len(later)==1 and len(test)==(0 if model=='forest12' else 1)
   rows.append(dict(variant=variant,model=model,loso_mean=loso.mean(),loso_std=loso.std(),delta_original=(loso-base).mean(),delta_delay1=(loso-prev).mean(),wins_vs_delay1=int(((loso-prev)>1e-12).sum()),later=later.iloc[0],test=test.iloc[0] if len(test) else np.nan,eligible=variant not in selection['excluded']))
 comparison=pd.DataFrame(rows);assert len(comparison)==27
 comparison.to_csv(P/'final_comparison.csv',index=False)
 ranked=comparison[(comparison.model=='forest20')&comparison.eligible].sort_values(['loso_mean','variant'],ascending=[False,True]);assert ranked.iloc[0].variant==selection['winner']
 eligible=list(ranked.variant);people=sorted(frame[frame.protocol=='loso'].fold.unique());stability=[]
 for held in people:
  s=frame[(frame.model=='forest20')&(frame.protocol=='loso')&(frame.fold!=held)&frame.variant.isin(eligible)].groupby('variant').session_mean.mean()
  winner=sorted(s.index,key=lambda v:(-s[v],v))[0];stability.append(dict(omitted=held,winner=winner))
 (P/'ranking_stability.json').write_text(json.dumps(stability,indent=2))
 verification=dict(new_jobs=203,control_jobs=58,officially_rescored_files=261,all_scores_match_official_code=True,first_stage_verification='verification.json',followup_verified_files=29,selected_variant=selection['winner'],selection_frozen_before_all_new_test=True,all_previous_artifacts_unchanged=True,all_followup_near_null_coordinates_zero=True,min_followup_scaler_variance=float(min(r['minimum_scaler_variance'] for r in records)),excluded_raw_variants=selection['excluded'],same_physical_trees_and_window_order=True,source_baseline_unchanged=True,summary_rows=27,fold_rows=261)
 (P/'final_verification.json').write_text(json.dumps(verification,indent=2))
 plot(comparison);report(comparison,verification,selection,stability)
 print(comparison.to_string(index=False));print(json.dumps(verification,indent=2));print(P/'FINAL_REPORT.md')


def plot(c):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Patch
 names=['Original','Delay1','Delay4','Stationary','Block raw*','Quadrature*','Path integral','Signature','Mixed halves']
 fig,axes=plt.subplots(1,2,figsize=(15,6));r=c[c.model=='forest20'].set_index('variant').loc[list(LABELS)]
 colors=['#667085','#0077b6','#2a9d8f','#52b788','#999999','#bbbbbb','#457b9d','#bc6c25','#e76f51']
 for ax,metric,title in [(axes[0],'loso_mean','LOSO: 8 people, sample SD'),(axes[1],'test','Public test: fixed configurations')]:
  bars=ax.bar(range(9),100*r[metric],color=colors,yerr=100*r.loso_std if metric=='loso_mean' else None,capsize=3)
  for i,v in enumerate(LABELS):
   if not r.loc[v,'eligible']:bars[i].set_hatch('///');bars[i].set_alpha(.5)
  ax.axhline(100*r.loc['delay1_oas',metric],color='#0077b6',ls='--',lw=1)
  ax.set(xticks=range(9),xticklabels=names,ylabel='Session-balanced accuracy (%)',ylim=(0,85),title=title)
  ax.tick_params(axis='x',rotation=40,labelsize=9)
 axes[0].legend(handles=[Patch(facecolor='gray',hatch='///',alpha=.5,label='Raw: null tangent features amplified by scaler')],fontsize=8)
 fig.suptitle('Replacing only R(X), with unchanged cumulative log-Euclidean pipeline\nNumerically eligible selection fixed before all new public tests',fontsize=12)
 fig.tight_layout();fig.savefig(P/'final_comparison.png',dpi=160);plt.close(fig)


def report(c,v,selection,stability):
 winner=selection['winner'];row=c[(c.variant==winner)&(c.model=='forest20')].iloc[0];prev=c[(c.variant=='delay1_oas')&(c.model=='forest20')].iloc[0]
 lag4=c[(c.variant=='lag4_oas')&(c.model=='forest20')].iloc[0]
 lines=['# Итог: замена R(X) при неизменном бейзлайне','',f"Заранее выбранный по forest20 LOSO численно допустимый вариант: **{LABELS[winner]}**. LOSO={row.loso_mean:.6f}, public test={row.test:.6f}. "
 f"Изменение к прежнему delay1: {100*(row.loso_mean-prev.loso_mean):+.2f} п.п. LOSO и {100*(row.test-prev.test):+.2f} п.п. test. "
 f"Улучшение LOSO у {int(row.wins_vs_delay1)}/8 людей. Выбор зафиксирован до всех новых test-оценок.",'',
 f'**Public test не подтвердил преимущество CV-лидера**: разность с delay1 {100*(row.test-prev.test):+.2f} п.п. Поэтому его нельзя объявлять общим оптимумом.','',
 f'Наиболее сбалансированный кандидат в ретроспективном сравнении фиксированных функций — **задержка 4**: LOSO={lag4.loso_mean:.6f}, test={lag4.test:.6f}; изменения к delay1 {100*(lag4.loso_mean-prev.loso_mean):+.2f} и {100*(lag4.test-prev.test):+.2f} п.п. Это интерпретация после просмотра test, а не новый ослеплённый выбор. Лаг 4 был зафиксирован до расчётов и после test не подбирался. Для подтверждения рекомендации нужен новый независимый набор.','',
 'Исследованы шесть литературно мотивированных функций и один корректирующий вариант после численной диагностики. Всего 203 новых оценки и 58 сохранённых контролей; все 261 файла предсказаний проверены официальными метриками. Исходные репозитории и бейзлайн не изменены.','',
 '## Сравнение','', 'Balanced accuracy, средняя по сессиям. LOSO — среднее ± выборочное SD восьми людей. Δ — разность с прежним лидером delay1. Звёздочка обозначает сырой результат функции с проблемными нулевыми координатами; он сохранён для прозрачности, но не участвует в численно допустимом ранжировании.','']
 for model,label in MODELS.items():
  lines += [f'### {label}','','| R(X) | LOSO | Δ к delay1, п.п. | Лучше delay1 | Поздняя сессия | Public test |','|---|---:|---:|---:|---:|---:|']
  for r in c[c.model==model].itertuples():
   test='—' if pd.isna(r.test) else f'{r.test:.6f}'
   lines.append(f'| {LABELS[r.variant]} | {r.loso_mean:.6f} ± {r.loso_std:.6f} | {100*r.delta_delay1:+.2f} | {r.wins_vs_delay1}/8 | {r.later:.6f} | {test} |')
  lines.append('')
 lines += ['![Итоговое сравнение](final_comparison.png)','','## Основной кандидат для следующей проверки: augmented covariance, lag4','','Для x(t)∈R^h строим z(t)=[x(t),x(t−4)] и R(X)=OAS(z). При 120 Гц это 33.333 мс; используются 476 совмещённых отсчётов. R имеет 82×82, как у предыдущей задержки 1, а дерево 20 видит 40×40. Те же формулы опорного логсреднего, scaler, C и физические электроды. Основание — [Carrara & Papadopoulo, 2023](https://arxiv.org/abs/2302.04508).','',
 '## CV-лидер: устойчивая функция временной ковариации','',
 'Разделим текущее 4-секундное окно на две половины по 240 отсчётов. После исходного полосового фильтра вычислим A=OAS(X₁), B=OAS(X₂), U=(A+B)/2, V=(A−B)/2. Затем', '',
 '$$R(X)=\\begin{pmatrix}U&V\\\\V&U\\end{pmatrix} = Q\\,\\operatorname{diag}(A,B)\\,Q^\\top,\\qquad Q=\\frac1{\\sqrt2}\\begin{pmatrix}I&I\\\\I&-I\\end{pmatrix}.$$', '',
 'Q ортогональна, поэтому собственные значения и обе исходные ковариации сохраняются, а SPD следует из SPD A и B. В коде координаты чередуются по физическому электроду. Матрица имеет размер 82×82; дерево с 20 электродами видит 40×40, как у прежнего delay1. Число окон, один шаг обновления на окно и исходное накопительное логсреднее не меняются. Матрицы A,B и преобразование не используют метки или другие окна.','',
 'Литературный источник временного разбиения — [Hussein et al., 2013, temporal hierarchy of covariance descriptors](https://www.ijcai.org/Proceedings/13/Papers/363.pdf). Упаковка в одну SPD-матрицу и ортогональная запись для данного бейзлайна — наша адаптация, не полное воспроизведение Cov3DJ.','',
 '## Почему сырой результат 62,03% не принят автоматически','',
 'В точной арифметике diag(A,B) порождает нулевые межблочные касательные координаты. На реальных данных eigendecomposition/logm оставляли ошибки округления:400 таких признаков имели дисперсии до 4.53×10⁻²⁷. Исходный StandardScaler превращал их в признаки с единичной дисперсией и значениями до 22.0. Поэтому высокая точность сырой блочной записи не была достаточным основанием объявить победу.','',
 'Перед test был зарегистрирован отдельный H7-протокол. Менялась только запись R(X) через фиксированную Q; сам scaler и исходный классификатор остались прежними. В минимальной проверке минимальная дисперсия стала 0.00194, а во всех 29 итоговых H7-оценках отсутствуют координаты с дисперсией ниже 10⁻²⁰. Численная диагностика: [block_numerics_probe.json](block_numerics_probe.json), [mixed_numerics_probe.json](mixed_numerics_probe.json).','',
 'Ортогональное преобразование сохраняет информацию, но покомпонентное масштабирование и регуляризация классификатора не обязаны быть инвариантны к нему. Поэтому разницу точности нельзя целиком приписать ошибкам округления. Мы показываем численно корректный вариант функции, а не причинную декомпозицию прироста. Квадратурная форма также имеет вынужденно нулевые собственные межквадратурные координаты и оставлена как сырой, нерекомендуемый вариант.','',
 '## Что дала литература и проверка непрерывности','',
 '- Задержанное вложение: [Carrara & Papadopoulo, 2023](https://arxiv.org/abs/2302.04508). Проверен lag4 при той же размерности, что lag1.',
 '- Стационарная автоковариация: [Wu & Xiao, 2011](https://www.stat.uchicago.edu/~wbwu/papers/cov-reviewJune15-2011-2.pdf) и [Hallac et al.,TICC](https://arxiv.org/abs/1706.03161). Заимствована структура, кластеризация и graphical lasso не добавлялись.',
 '- Комплексные ковариации: [von Sachs, 2020](https://doi.org/10.1146/annurev-statistics-031219-041138). Проверена оконная квадратурная адаптация, без замены классификатора.',
 '- Непрерывный путь и упорядоченные интегралы: [Király & Oberhauser, 2019](https://jmlr.org/papers/v20/16-314.html). Проверены предложенные здесь интегральная ковариация линейного интерполянта и OAS+локальный сигнатурный Gram.','',
 'Точная интегральная ковариация вычислялась двумя Gauss-узлами на каждом линейном интервале, без прореживания. Это явная модель непрерывного пути между отсчётами. Она остаётся 41×41. Сигнатурная функция также 41×41; однако её Gram-проекция теряет полное направление времени: обращение времени ортогонально меняет глубину 2 и оставляет Gram прежним. Это проверено отдельно без меток и без изменения фильтрации в модельных экспериментах. Гарантии полного сигнатурного ядра не переносятся на такую компактную упаковку.','',
 'Ни непрерывность, ни свойство SPD сами по себе не обеспечивают хорошего распознавания. Результаты относятся к конкретным функциям; они не доказывают бесполезность производных, других интерполянтов, более глубоких сигнатур или других лагов. Временное разбиение позволяет сохранить различие локальных статистик, но отдельного доказательства, что именно абсолютный порядок половин вызвал прирост, этот эксперимент не даёт.','',
 'Полный обзор девяти первичных работ: [literature/survey.md](literature/survey.md). Точные формулы первых шести функций и их полные сырые результаты: [REPORT.md](REPORT.md).','',
 '## Проверки и пределы вывода','',
 '- Официальным кодом перепроверены 261 файл и четыре метрики, atol=1e−14; сохранены те же люди, окна, деревья, полосы, C=.01,seed0 и prior_weight5.',
 '- Все предыдущие результаты и исходные репозитории неизменны. Для test обучение только на исходных шести train-людях.',
 '- 101 тест проверяет исходный код, прошлые функции, новые формулы, SPD, потоковость, кэш и ортогональную запись; полный лог — logs/final-tests.log.',
 '- 36 базовых наборов кэша содержат 95 580 проверенных SPD-матриц. H7 применяет ортогональную конгруэнцию к тем же матрицам, без нового обучения или межоконных статистик.',
 '- Выбор максимума среди нескольких функций на восьми dev-людях исследовательский и оптимистичен. Новый test не использовался для выбора, но public test уже был просмотрен в прошлых сериях. Закрытая независимая оценка остаётся необходимой для окончательной проверки.',
 f"- При исключении одного человека из усреднения тот же лидер сохраняется в {sum(r['winner']==winner for r in stability)}/8 случаев. Это описательная устойчивость ранжирования, не вложенная кросс-валидация.",
 '- Один seed, фиксированные настройки функций. Ранжирование относится к LOSO; общий оптимум всех SPD-представлений не установлен.',
 '- Время fit использует готовый кэш; это не полная задержка приложения. Все функции используют только доступное текущее окно, а исходная память модели и границы сессий сохраняются.','',
 '## Артефакты','',
 '[final_comparison.csv](final_comparison.csv), [final_fold_scores.csv](final_fold_scores.csv), [final_verification.json](final_verification.json), [followup_mixed/selection.json](followup_mixed/selection.json), [followup_mixed/PROTOCOL.md](followup_mixed/PROTOCOL.md), [followup_mixed/mixed_features.py](followup_mixed/mixed_features.py). Команды исходных стадий — в [README.md](README.md).','']
 (P/'FINAL_REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
