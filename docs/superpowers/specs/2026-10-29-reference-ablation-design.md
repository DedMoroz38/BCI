# Reference matrix ablation design

Request: repeat the previously reproduced EEG experiment with several reference-matrix choices from the supplied discussion.

## Controlled comparison

Only causal session reference updating changes. All variants start at the **same original training log-Euclidean prior** M0=exp(mean(log(R_train))), with prior weight w0=5. This deliberately isolates the session update rule; it is not a replication of full EA/REBIAS literature pipelines and does not change the training-prior estimator.

Fixed configurations before computing new scores:

1. Original cumulative log-Euclidean (existing verified control): L_t=(5 L0+sum(log R_i))/(5+t).
2. `euclid_cumulative`: M_t=(5 M0+sum R_i)/(5+t).
3. `riemann_recursive`: M_t=M_{t-1} #_{1/(5+t)} R_t, where # is affine-invariant geodesic interpolation. This is an inductive/recursive estimator, not the exact batch Fréchet mean in the general noncommuting case.
4. `logeuclid_ema`: S_t=0.95 S_{t-1}+log R_t, n_t=0.95 n_{t-1}+1, L_t=S_t/n_t, S_0=5 L0,n_0=5. This is a normalized exponential window-weighting rule that preserves the original prior mass at initialization. Its gain tends to 0.05; it is not constant-gain EMA during warm-up. Window-weight half-life=log(.5)/log(.95)=13.51 windows.

Features remain vec_sqrt2(log(M_t^-1/2 R_t M_t^-1/2)); the current window updates M_t before classification, as in the original. Training-session transformations use the same chosen updater as inference. StandardScaler is refitted on those changed training features. Covariances, filters, seeds, tree selections, bootstraps, logistic C=0.01 and all other machinery are unchanged.

## Protocol and outputs

Same PROTEUS prepared files/hashes, 6/2/2 people and 1521/379/755 windows. Per variant: 24 LOSO + 3 later-session + 2 public-test evaluations, 29 jobs total; 87 new jobs across three candidates. Baseline control reuses the 29 verified original results. Test fits only six training subjects and covers logreg and forest20; forest12 test is not added. Alpha/decay, model parameters and variants will not be selected by test score.

Save every prediction, fold score, chosen tree band/channel list, source/environment/data provenance. Re-score all predictions using official competition metrics. Compare paired LOSO deltas per person, not independent window-level replicates. Show sample SD and win/loss counts; results remain exploratory with eight dev subjects, one later-session subject and two public-test subjects.

## Implementation approaches

- Chosen: an external subclass overrides `new_session`, injected into the two unchanged upstream model factories inside a scoped context. This preserves exact upstream training/randomization logic and source checkouts, with isolated results under reproduction/reference_ablation.
- Rejected: copy/rewrite the entire forest; it risks accidental changes in sample weighting or RNG ordering.
- Not used: exact batch Riemannian mean recomputed on every prefix, rest-based references or median. These require a different cost/protocol/data-selection design and are not needed for this requested several-variant comparison.

## Verification

Analytic scalar/diagonal mean checks; noncommuting two-matrix AIRM reference against pyRiemann; positive definiteness; exponential weight formula and recency; zero-future dependence, fresh-session reset, and batch/stream prediction parity. Control adapter must equal original on small data exactly; identical tree selections must be checked against corresponding stored control jobs. Upstream sources and original results must remain unchanged.
