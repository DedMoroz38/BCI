# CPU synergy results

Primary: first-session cell BA; mean repeat metric per person, then equal mean over 10 people.

                             model  first_session_ba   worst3  first_nll  all_session_cell_ba  first20_ba  middle21_60_ba  tail61_ba  mean_repeat_sd  folds
       lag4_nuclear005_50__balance          0.583194 0.430466   0.980065             0.563900    0.609603        0.592439   0.577568        0.008384     10
mssa8_mix25_nuclear005_50__balance          0.582907 0.427806   0.980881             0.564192    0.609603        0.598954   0.575169        0.008669     10
mssa8_mix25_nuclear001_50__balance          0.582808 0.431017   0.954360             0.562206    0.595437        0.579923   0.582300        0.010208     10
       lag4_nuclear001_50__balance          0.580717 0.428383   0.953823             0.562110    0.588194        0.579745   0.580425        0.009710     10
         mssa8_mix25_lr50__balance          0.576352 0.433391   0.950723             0.557893    0.579861        0.571357   0.577644        0.007154     10
        lag4_matched_lr50__balance          0.575936 0.433419   0.950189             0.557178    0.583333        0.573166   0.576171        0.008716     10
                lag4_nuclear001_50          0.575365 0.426376   0.955743             0.549412    0.565020        0.582867   0.574797        0.006895     10
         mssa8_mix25_nuclear001_50          0.574844 0.425480   0.956320             0.547552    0.573234        0.577244   0.573984        0.008525     10
                lag4_nuclear005_50          0.573545 0.409300   0.980672             0.549693    0.585734        0.580859   0.569880        0.008274     10
         mssa8_mix25_nuclear005_50          0.571653 0.409244   0.981514             0.547741    0.583948        0.579190   0.568346        0.008857     10
                  mssa8_mix25_lr50          0.570790 0.436792   0.953088             0.545232    0.544226        0.573024   0.573957        0.012209     10
                 lag4_matched_lr50          0.569767 0.436820   0.952507             0.545188    0.542976        0.574033   0.572435        0.011722     10

Runtime: 323.5 s; failed folds: 0; failed fold/head configurations: 0.

Exploratory comparison on reused validation participants; no independent test claim. Protocol frozen in PROTOCOL.md.
