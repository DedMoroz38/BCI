"""Independent probability rescoring of the fixed quick screening."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent

def scores(f):
    y=f['y'];p=f['proba'];pred=p.argmax(1)
    np.testing.assert_array_equal(pred,f['pred'])
    stream=np.char.add(np.char.add(f['subject'],'/'),f['session'])
    def ba(idx):
        a,b=y[idx],pred[idx]
        return np.mean([np.mean(b[a==c]==c) if np.any(a==c) else 0 for c in np.union1d(a,b)])
    session=np.mean([ba(np.flatnonzero(stream==s)) for s in np.unique(stream)])
    cell=np.mean([ba(np.flatnonzero((stream==s)&(f['context']==c))) for s,c in set(zip(stream,f['context']))])
    return dict(session_mean=float(session),cell_mean=float(cell),accuracy=float(np.mean(y==pred)),
                pooled=float(ba(np.arange(len(y)))),log_loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()),
                brier=float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))))

def main():
    rows=[];manifest=hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest()
    for p in sorted((HERE/'results').glob('*/*.json')):
        if p.stem not in ('baseline','mixed50','stretch50','balance05','balance10'):continue
        r=json.loads(p.read_text());pred=p.with_suffix('.npz')
        assert r['manifest']==manifest
        assert hashlib.sha256(pred.read_bytes()).hexdigest()==r['prediction_sha256']
        with np.load(pred) as f:
            for k,v in scores(f).items():np.testing.assert_allclose(v,r[k],rtol=0,atol=1e-12)
        r['protocol']='loso' if r['tag'].startswith('loso_') else 'later';rows.append(r)
    assert len(rows)==45,len(rows)
    df=pd.DataFrame(rows);df.to_csv(HERE/'fold_scores.csv',index=False)
    base=df[(df.variant=='baseline')&(df.protocol=='loso')].set_index('tag')
    summary=[];paired={}
    for name in ('baseline','mixed50','stretch50','balance05','balance10'):
        cv=df[(df.variant==name)&(df.protocol=='loso')].set_index('tag').reindex(base.index)
        assert len(cv)==8 and not cv.session_mean.isna().any()
        later=df[(df.variant==name)&(df.protocol=='later')].iloc[0]
        delta=(cv.session_mean-base.session_mean).to_numpy()
        sample=np.random.default_rng(0).integers(8,size=(20000,8))
        low,high=np.quantile(delta[sample].mean(1),[.025,.975])
        summary.append(dict(variant=name,loso_session=float(cv.session_mean.mean()),loso_cell=float(cv.cell_mean.mean()),
                            later_session=float(later.session_mean),later_cell=float(later.cell_mean),
                            loso_log_loss=float(cv.log_loss.mean()),loso_brier=float(cv.brier.mean()),
                            delta=float(delta.mean()),ci_low=float(low),ci_high=float(high),wins=int(np.sum(delta>0))))
        paired[name]={tag:float(v) for tag,v in zip(base.index,delta)}
    table=pd.DataFrame(summary);table.to_csv(HERE/'comparison.csv',index=False)
    (HERE/'paired.json').write_text(json.dumps(paired,indent=2))
    timing=[json.loads(p.read_text()) for p in (HERE/'results').glob('*/timing.json')]
    assert len(timing)==9 and all(r['refit_baseline_exact'] for r in timing)
    count=sum(len(r['warnings']) for r in timing)
    lines=['# Быстрые локальные эксперименты','',
           '8LOSO-разбиений по людям и одна поздняя сессия.45 оценок независимо пересчитаны по сохранённым предсказаниям.',
           'Все варианты имеют50экспертов. Гиперпараметры зафиксированы заранее,публичный тест не запускался.',
           'Все9 переобучений исходного Lag4 совпали по предсказаниям с архивом; вероятности каждого эксперта совпали с точностью1e-7.','',
           '| Вариант | LOSO session BA | LOSO cell BA | Поздняя session BA | Δ LOSO | Победы |',
           '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['variant']} | {100*r['loso_session']:.2f}% | {100*r['loso_cell']:.2f}% | {100*r['later_session']:.2f}% | {100*r['delta']:+.2f} п.п. | {r['wins']}/8 |")
    lines+=['','mixed50:17Lag4+17обычныхOAS+16адаптивныхлагов; физические каналы и бутстрап каждого эксперта сохранены.',
            'stretch50:накопительная нормализация RMS касательных признаков,prior10,одинаковая при обучении и инференсе.',
            'balance05/10:коррекция предсказанных частот классов со степенью.5/1 иprior10,сброс по сессиям.',
            f'Сумма CPU-времени9разбиений:{sum(r["total_seconds"] for r in timing):.1f}с; предупреждений:{count}.',
            'Интервалы в comparison.csv — описательный парный бутстрап8людей; обучающие выборки фолдов пересекаются.',
            'Это исследовательский скрининг на ранее использовавшихся людях. Поздняя сессия представлена одним участником.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (HERE/'verification.json').write_text(json.dumps(dict(evaluations=45,baseline_refits=9,
        baseline_prediction_matches=True,independent_rescore=True,manifest=manifest,warnings=count),indent=2))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
