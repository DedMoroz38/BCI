# CPU synergy outcome

The frozen mSSA + nuclear combination improves its matched mSSA LR comparator but does not improve the strongest ordinary Lag4 nuclear comparator on the primary metric.

Metric: first-session cell balanced accuracy; average the two held-out repeat metrics within each person, then take an equal mean over ten people. All models use the exact fixed ten 8-person/2-person folds.

| Model | Primary BA |
|---|---:|
| Existing Lag4 nuclear .05 + causal balance | 58.3193807162% |
| New mSSA rank8 mix25 nuclear .05 + causal balance | 58.2907010876% |
| New mSSA rank8 mix25 nuclear .01 + causal balance | 58.2808212829% |
| Existing Lag4 nuclear .01 + causal balance | 58.0716536893% |
| Existing mSSA rank8 mix25 LR50 + causal balance | 57.6352119639% |

The new .05 model gains 0.6554891237 percentage points over mSSA LR and loses 0.0286796286 points against the best ordinary Lag4 model. The new .01 model gains 0.2091675936 points over the same-tau Lag4 comparator (8 people better, 1 tied, 1 worse); it remains 0.0385594333 points below the strongest .05 Lag4 comparator. These small differences do not establish a new SOTA.

Without balance: new nuclear .01 = 57.4844387282%; new nuclear .05 = 57.1653090625%; existing mSSA LR = 57.0789611914%; existing Lag4 nuclear .01 = 57.5364660580%.

Runtime: 323.466087583 seconds (5 minutes 23.47 seconds), two CPU workers, one native thread each. Ten folds complete; zero failed folds, zero failed head configurations, zero artifact verification errors. Saved 500 expert states containing 1000 new fitted heads and 40 per-window prediction files (two taus, plain/balanced, ten folds). All four causal/leakage/objective tests pass.

`comparison.csv` contains all twelve complete new/comparator configurations and secondary metrics; `per_person.csv` and `per_person_repeat.csv` contain the specified aggregation detail. `paired_comparisons.csv` and `paired_per_person.csv` contain matched descriptive differences. `STATUS.json`, `ARTIFACT_VERIFICATION.json`, `tests.log`, and `execution.log` provide completion and verification evidence. `PROTOCOL.md` was frozen before any new score; `SOURCE_AUDIT.md` documents source-only reuse and inference chronology.

This is an exploratory result on repeatedly used validation participants. There was no target-dependent fitting, additional post-score tuning, baseline refit, remote execution or submission.
