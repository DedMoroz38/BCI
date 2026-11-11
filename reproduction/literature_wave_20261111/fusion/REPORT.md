# Fixed full-ensemble forest substitutions

Same saved two-seed CNN forecasts, fixed inherited TTA, no fitting or weight tuning. Raw forest probabilities replaced before TTA. Primary: equal-person first-session BA, averaging two repeat metrics per person.

 scheme                       model  first_session_ba   worst3  median_person  all_session_cell_ba  first_nll  first20_ba  middle21_60_ba  tail61_ba  mean_repeat_sd  folds  delta_first_ba_pp  delta_first_nll
pair8_2    full_6458_fixed30_recipe          0.637010 0.471903       0.652920             0.617522   0.853461    0.589107        0.650770   0.643922        0.013501     10           0.000000         0.000000
pair8_2      full_nuclear001_recipe          0.632036 0.463221       0.650358             0.614530   0.855775    0.582262        0.647824   0.638869        0.012610     10          -0.497424         0.002314
pair8_2 full_mssa_nuclear001_recipe          0.632009 0.459692       0.654214             0.614919   0.856372    0.587024        0.649163   0.637713        0.013336     10          -0.500118         0.002911
pair8_2      full_nuclear005_recipe          0.626939 0.476325       0.650526             0.606750   0.870565    0.572282        0.657865   0.630913        0.009634     10          -1.007056         0.017104
pair8_2 full_mssa_nuclear005_recipe          0.625300 0.474561       0.653130             0.605759   0.871532    0.574365        0.656995   0.628585        0.009156     10          -1.170979         0.018071

Exploratory validation on previously studied public participants; no independent hidden-test claim. Source and target metadata audit: INPUT_AUDIT.json.
