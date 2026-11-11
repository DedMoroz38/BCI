# Extra input policies — declared after first spectral results

First complete spectral pass had best52.24% OOF versus standalone balanced
Lag4 forest58.53%. Input EA + causal log-power recentering was one unproven
choice, so negative results do not distinguish representation from alignment.
Before looking at any extra-input result, define three additional input
policies; keep the identical11 fixed heads/ranks for every policy:

1. spectral_source:causal input EA, no causal recentering of log-power.
2. spectral_center:no EA, causal perchannel/frequency log-power recentering.
3. spectral_relative:no EA,no session recentering; subtract one global
   log-power mean per current window,removing its overall amplitude only.

The original spectral pass remains separately labeled. All policies use
outer-source weighted feature standardization and equal-person/class weights.
Same10LOSO +later_last, no parameter fitting on outer participants. These33
additional model configurations are adaptive exploration, not blind validation.

One additional CPU worker while the two forest workers run:3processes total,
1BLAS thread each. Outputresults_spectral_inputs kept separate to avoid two
independent report writers racing; predictions can be summarized jointly later.
No GPU or CNN. Overall study now54 learned model configurations across11folds.
