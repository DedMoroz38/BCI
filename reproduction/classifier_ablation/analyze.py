"""Independently verify all70 jobs and summarize paired subject-level effects."""
import csv,itertools,json,sys,importlib.util
import numpy as np
from common import HERE,ROOT,OUT,sha,write_json
from classifiers import VARIANTS
# Upstream also contains run.py and common prepends that directory to sys.path.
# Load this experiment's runner by exact path in a fresh analysis process.
_spec=importlib.util.spec_from_file_location('classifier_ablation_run',HERE/'run.py')
_runner=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_runner)
config_hash=_runner.config_hash
sys.path.insert(0,str(ROOT/'2026-competition'))
from benchmark_utils.metrics import balanced_accuracy,group_scores

LABELS=dict(baseline='Logistic regression (submitted baseline)',lda_lw='Shrinkage LDA',
            svm_linear='Linear SVM + grouped calibration',svm_rbf='RBF SVM + grouped calibration',
            ridge='Ridge + grouped calibration',elasticnet='Elastic net + grouped calibration',
            mlp='MLP32 (PyTorch CUDA)')

def probability_scores(y,p):
    """Manual probability metrics,independent of run.py/sklearn's log_loss."""
    eps=np.finfo(p.dtype).eps
    return dict(log_loss=float(-np.log(np.clip(p[np.arange(len(y)),y],eps,1-eps)).mean()),
                brier=float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))))


def selection_stability(scores,subjects,samples):
    """Descriptive paired resampling of existing person scores; no model refits."""
    names=sorted(scores);matrix=np.array([scores[name] for name in names]);n=len(subjects)
    assert matrix.shape==(len(names),n) and n>=2
    omitted=[]
    for j,subject in enumerate(subjects):
        means=np.delete(matrix,j,axis=1).mean(1)
        omitted.append(dict(omitted_subject=subject,winner=names[int(means.argmax())]))
    winners=np.argmax(matrix[:,samples].mean(2),axis=0)
    return dict(leave_one_subject_out=omitted,
                leave_one_subject_out_wins={name:sum(r['winner']==name for r in omitted) for name in names},
                paired_bootstrap_winner_fraction={name:float(np.mean(winners==i)) for i,name in enumerate(names)},
                bootstrap_samples=len(samples),tie_break='alphabetical',
                interpretation='Descriptive resampling stability of8 observed people; not independent-cohort evidence or probability of true superiority')


def main():
    records=[];by={};feature_hashes={};baseline_matches=0
    for name in VARIANTS:
        by[name]={}
        for path in sorted((OUT/name).glob('*.json')):
            r=json.loads(path.read_text());tag=r['tag'];predfile=path.with_name(tag+'_predictions.npz')
            assert r['classifier_config_sha256']==config_hash(),path
            assert sha(predfile)==r['predictions_sha256'],path
            with np.load(predfile) as f:
                y,pred,p,trees=f['y'],f['pred'],f['proba'],f['tree_proba']
                assert trees.shape==(50,len(y),3) and np.isfinite(trees).all() and trees.min()>=0
                np.testing.assert_allclose(trees.sum(-1),1,atol=1e-6)
                expected=trees.mean(0);expected/=expected.sum(1,keepdims=True)
                np.testing.assert_allclose(p,expected,atol=1e-14,rtol=0)
                np.testing.assert_array_equal(pred,p.argmax(1))
                stream=np.char.add(np.char.add(f['subject'],'/'),f['session'])
                cells=np.stack([stream,f['context']],axis=1)
                metrics=dict(session_mean=float(group_scores(balanced_accuracy,y,pred,stream).mean()),
                             cell_mean=float(group_scores(balanced_accuracy,y,pred,cells).mean()),
                             pooled=balanced_accuracy(y,pred),accuracy=float(np.mean(y==pred)))
                metrics.update(probability_scores(y,p))
                for key,val in metrics.items():assert np.isclose(val,r[key],atol=1e-14,rtol=0),(path,key)
                reference=ROOT/'temporal_spd_research/results/lag4_oas'/f"forest20_{r['protocol']}_{r['fold'].replace('/','_')}_predictions.npz"
                with np.load(reference) as old:
                    for key in ['y','subject','session','context','run','onset']:np.testing.assert_array_equal(f[key],old[key])
                    if name=='baseline':np.testing.assert_array_equal(pred,old['pred']);baseline_matches+=1
                if r['protocol']!='later':assert not set(r['train_subjects'])&set(r['eval_subjects'])
            feature_hashes.setdefault(tag,set()).add(r['feature_manifest_sha256'])
            by[name][tag]=r;records.append(r)
        assert len(by[name])==10,(name,len(by[name]))
    assert len(records)==70 and baseline_matches==10
    assert all(len(v)==1 for v in feature_hashes.values())
    selection=json.loads((OUT/'selection.json').read_text())
    for name,digest in selection['cv_result_sha256'].items():assert sha(OUT/name)==digest
    assert len(selection['cv_result_sha256'])==63
    loso_tags=sorted(t for t in by['baseline'] if t.startswith('loso_'))
    cv_means={name:float(np.mean([by[name][t]['session_mean'] for t in loso_tags])) for name in VARIANTS}
    for name,value in cv_means.items():
        assert np.isclose(value,selection['scores'][name],atol=1e-14,rtol=0),(name,'frozen CV score')
    assert selection['winner']==sorted(cv_means,key=lambda name:(-cv_means[name],name))[0]
    rng=np.random.default_rng(0);samples=rng.integers(0,8,size=(20000,8));paired=[];rows=[]
    signs=np.array(list(itertools.product([-1,1],repeat=8)))
    for name in VARIANTS:
        cv=np.array([by[name][t]['session_mean'] for t in loso_tags])
        base=np.array([by['baseline'][t]['session_mean'] for t in loso_tags]);delta=cv-base
        low,high=np.quantile(delta[samples].mean(1),[.025,.975])
        test=by[name]['test_official'];later=by[name]['later_last']
        row=dict(variant=name,label=LABELS[name],loso_session_mean=float(cv.mean()),
                 loso_std_subjects=float(cv.std(ddof=1)),loso_delta=float(delta.mean()),
                 paired_bootstrap95_low=float(low),paired_bootstrap95_high=float(high),
                 loso_subject_wins=int((delta>0).sum()),loso_subject_ties=int((delta==0).sum()),
                 later_session_mean=later['session_mean'],test_cell_mean=test['cell_mean'],
                 test_session_mean=test['session_mean'],test_pooled=test['pooled'],test_accuracy=test['accuracy'],
                 test_log_loss=test['log_loss'],test_brier=test['brier'],
                 total_classifier_seconds=sum(r['classifier_seconds'] for r in by[name].values()),
                 convergence_warnings=sum(r['convergence_warning_count'] for r in by[name].values()))
        if name!='baseline':
            row['exploratory_signflip_p']=float(np.mean(np.abs((signs*delta).mean(1))>=abs(delta.mean())-1e-15))
        else:row['exploratory_signflip_p']=1.
        rows.append(row)
        paired += [dict(variant=name,subject=t,baseline=by['baseline'][t]['session_mean'],score=by[name][t]['session_mean'],delta=by[name][t]['session_mean']-by['baseline'][t]['session_mean']) for t in loso_tags]
    ordered=sorted([r for r in rows if r['variant']!='baseline'],key=lambda r:r['exploratory_signflip_p'])
    adjusted=0
    for i,r in enumerate(ordered):
        adjusted=max(adjusted,min(1,(6-i)*r['exploratory_signflip_p']));r['exploratory_holm_p']=adjusted
    rows[0]['exploratory_holm_p']=1.
    stability=selection_stability({name:[by[name][t]['session_mean'] for t in loso_tags] for name in VARIANTS},loso_tags,samples)
    write_json(OUT/'selection_stability.json',stability)
    for row in rows:
        name=row['variant'];row['leave_one_subject_out_selection_wins']=stability['leave_one_subject_out_wins'][name]
        row['paired_bootstrap_selection_fraction']=stability['paired_bootstrap_winner_fraction'][name]
    for filename,values in [('comparison.csv',rows),('paired_subject_scores.csv',paired)]:
        with (OUT/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    write_json(OUT/'verification.json',dict(verified_jobs=70,exact_baseline_prediction_jobs=10,
               common_feature_manifests=True,official_metric_agreement=True,
               independently_recomputed_log_loss_and_brier=True,
               cv_selection_frozen_before_public_test=True,cv_records_locked=63,
               classifier_config_sha256=config_hash(),selection=selection,
               warning_counts={r['variant']:r['convergence_warnings'] for r in rows}))
    def pct(x):return f'{100*x:.2f}'
    lines=['# Classifier-only ablation of the submitted Lag4 forest','',
           'All70 evaluations completed. Geometry,scalers,tree draws and subject bootstraps are shared. '
           'The original LR reproduced all10 prior baseline prediction files exactly.','',
           f"LOSO-selected model,chosen before new public-test scores: **{selection['winner']}**.",'',
           '| Classifier | LOSO session BA mean ± SD,% | Delta,pp | Later session,% | Public test cell BA,% | Public test session BA,% |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['label']} | {pct(r['loso_session_mean'])} ± {pct(r['loso_std_subjects'])} | {100*r['loso_delta']:+.2f} | {pct(r['later_session_mean'])} | {pct(r['test_cell_mean'])} | {pct(r['test_session_mean'])} |")
    winner=selection['winner']
    lines += ['', '## Selection stability','',
              f"The selected {winner} remains the winner in {stability['leave_one_subject_out_wins'][winner]}/8 leave-one-person-out reaggregations and in {100*stability['paired_bootstrap_winner_fraction'][winner]:.1f}% of20,000 paired person bootstraps. "
              'These reaggregate existing LOSO scores without retraining. They describe sensitivity to the eight observed people; they are not probabilities of superiority on a new cohort. Full counts are in selection_stability.json.']
    lines += ['', '## Interpretation limits','',
              'Fixed literature-informed configurations were compared,not optimally tuned model families. '
              'There are only8 LOSO people; their training folds overlap. Paired bootstrap intervals '
              'and sign-flip/Holm values in the CSV are exploratory descriptions,not independent-cohort '
              'confirmation. The later-session estimate is one held-out session. Public test comprises '
              'two people and has already been inspected in prior research. It is not a sealed score.',
              '', 'The probability interface for SVM,ridge and elastic net includes subject-grouped '
              'sigmoid calibration. This is part of the classifier replacement. Geometry is frozen '
              'at the outer-training fold during inner calibration; no outer-evaluation labels enter fit.',
              '', 'MLP uses300 fixed full-batch Adam epochs on CUDA,not the provisional local sklearn '
              'L-BFGS fixture. Its fixed-epoch results do not assert convergence. All numerical '
              'ConvergenceWarnings from classical classifiers remain in individual job JSON files.',
              '', 'Evidence tiers and adaptations are documented in ../literature/survey.md. '
              'Ridge has direct fMRI tangent-connectome support; RBF has a weaker direct EEG source. '
              'The strongest direct EEG precedents are TS-LDA,TS-linear-SVM and TS-elastic net.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,5))
        names=[r['variant'] for r in rows];xx=np.arange(len(rows));width=.36
        ax.bar(xx-width/2,[100*r['loso_session_mean'] for r in rows],width,label='LOSO: mean session BA')
        ax.bar(xx+width/2,[100*r['test_cell_mean'] for r in rows],width,label='Public test: mean cell BA')
        ax.axhline(100/3,color='grey',ls=':',lw=1);ax.set_ylabel('Balanced accuracy,%')
        ax.set_xticks(xx,names,rotation=20);ax.legend();fig.tight_layout()
        fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    except ImportError:print('matplotlib unavailable; CSV and verified report saved',flush=True)
    print(json.dumps(rows,indent=2),flush=True)
    print('VERIFIED ALL70 JOBS',flush=True)

if __name__=='__main__':main()
