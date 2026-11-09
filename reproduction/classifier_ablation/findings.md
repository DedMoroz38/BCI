# Final findings — fixed classifier comparison completed

All70 evaluations completed on Kaggle with the submitted Lag4 geometry,50 trees,
the original bootstrap/channel draws,and seven predeclared classifier settings.
Independent Kaggle and local analysis verified every prediction file,all10 exact
baseline label matches,the63 frozen CV result hashes,and all probability metrics.
Training integrity tests17/17 and local suite20/20 passed. Sources and numerical
versions are recorded; no representation or test-driven hyperparameter changes.

Linear SVM is the predeclared LOSO winner:57.5186% mean session BA versus56.2801%
for LR,+1.2385pp. Its paired exploratory95% interval[-0.8644,+3.3553]pp includes
zero. SVM wins5/8 people; it stays the winner after omitting any one person's
score,and wins69.705% of paired subject bootstraps. These are descriptive
stability results from eight people with overlapping training folds.

That CV gain does not transfer to the public test: SVM50.4447% cell BA versus
LR55.7164%,-5.2716pp. Every fixed alternative is below LR on this metric. LDA is
closest at55.2826% and improves probability quality slightly: logloss0.98219 vs
0.99296 and Brier0.58919 vs0.59619. Thus the preferred head depends on the metric;
there is no demonstrated public-test BA gain from this classifier-only change.

The fixed MLP scores54.5185% LOSO,73.3755% later-session and53.6039% public-test
cell BA. It wins1/8 LOSO people versus LR. The observed within-person-session
benefit does not establish an unseen-person benefit. Its300 fixed epochs are
not a convergence guarantee. This does not rank optimally tuned MLP families.

ElasticNet emits2423 ConvergenceWarnings and scores52.5757% LOSO/45.8485% test.
Its result is qualified by the10000-iteration optimizer limit; do not interpret
it as the performance of a fully converged or tuned elastic-net family. It is
also the CPU bottleneck (~31748 summed classifier seconds across concurrent
jobs),while the batched CUDA MLP takes~32 summed seconds for all10 folds.
Those observed timings are not a controlled hardware-efficiency benchmark.

The public test has two people and was seen in earlier research. No sealed-score
or independent-cohort confirmation claim is justified. No new submission made.
Verified tables,report and audit summaries are in to_human/; all raw predictions
remain in results/ and the immutable Kaggle download. Literature evidence tiers
and classifier-only adaptations remain explicit in literature/survey.md.
