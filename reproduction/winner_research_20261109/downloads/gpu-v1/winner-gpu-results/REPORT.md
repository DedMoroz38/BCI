# Winner-inspired research

Completed prediction records: 462. Complete LOSO configurations: 42.

Historical full baseline: OOF62.7234%, later71.1102%. No baseline training or parity gates.
Public CV is exploratory; new branches are also evaluated in the full ensemble.

                                         name  oof_cell  mean_person  later_cell      nll    brier   delta_pp
          latent_meanstd_p10_mean2__late_w0.2  0.632294     0.657617    0.716395 0.858086 0.504782   0.506006
             latent_mean_p10_mean2__late_w0.2  0.629659     0.653612    0.711102 0.865178 0.508873   0.242560
             latent_mean_p10_mean2__late_w0.1  0.629530     0.651711    0.705726 0.862079 0.507541   0.229663
          latent_meanstd_p10_mean2__late_w0.1  0.628793     0.652062    0.705726 0.858471 0.505529   0.155966
          latent_deepset_p10_mean2__late_w0.1  0.626944     0.648792    0.705726 0.867275 0.510684  -0.028934
       tsmnet_causal_le_lag4_mean2__late_w0.1  0.626739     0.650721    0.710865 0.857592 0.505797  -0.049523
          latent_deepset_p40_mean2__late_w0.1  0.626188     0.648092    0.705726 0.867044 0.510548  -0.104538
       tsmnet_causal_le_lag4_mean2__late_w0.2  0.625313     0.647174    0.705657 0.855838 0.505194  -0.192115
       latent_meanstd_p10_mean2__lag4_mix_tta  0.624895     0.650253    0.700686 0.860291 0.507362  -0.233864
          latent_deepset_p10_mean2__late_w0.2  0.624683     0.647759    0.703275 0.874938 0.514924  -0.255040
          latent_deepset_p40_mean2__late_w0.2  0.624671     0.647742    0.703275 0.874480 0.514652  -0.256240
            tsmnet_causal_le_mean2__late_w0.1  0.624558     0.646556    0.708498 0.859345 0.507066  -0.267609
          latent_mean_p10_mean2__lag4_mix_tta  0.618617     0.644466    0.711270 0.868007 0.512466  -0.861630
    tsmnet_causal_le_lag4_mean2__lag4_mix_tta  0.618535     0.639859    0.685431 0.864974 0.512211  -0.869838
            tsmnet_causal_le_mean2__late_w0.2  0.618501     0.641600    0.690228 0.859032 0.507538  -0.873228
         tsmnet_causal_le_mean2__lag4_mix_tta  0.613878     0.636157    0.671726 0.871413 0.516666  -1.335537
       latent_deepset_p40_mean2__lag4_mix_tta  0.613073     0.638395    0.692853 0.874247 0.516677  -1.416057
       latent_deepset_p10_mean2__lag4_mix_tta  0.613048     0.637666    0.692853 0.874835 0.517075  -1.418537
   latent_meanstd_p10_mean2__lag4_mix_balance  0.604667     0.630769    0.679511 0.963224 0.575754  -2.256684
   latent_deepset_p40_mean2__lag4_mix_balance  0.596415     0.620219    0.669095 0.975421 0.583757  -3.081872
   latent_deepset_p10_mean2__lag4_mix_balance  0.596075     0.619713    0.669095 0.975685 0.583940  -3.115886
tsmnet_causal_le_lag4_mean2__lag4_mix_balance  0.595824     0.613577    0.669025 0.958884 0.573277  -3.141022
      latent_mean_p10_mean2__lag4_mix_balance  0.595246     0.620332    0.682289 0.967606 0.578676  -3.198793
     tsmnet_causal_le_mean2__lag4_mix_balance  0.594054     0.613065    0.684343 0.959903 0.573983  -3.317931
                     latent_meanstd_p10_mean2  0.559364     0.597440    0.575018 0.991382 0.575715  -6.786951
                        latent_mean_p10_mean2  0.540973     0.574639    0.581467 1.035795 0.598430  -8.626025
                     latent_meanstd_p10_seed0  0.528534     0.561093    0.597156 1.061074 0.607592  -9.870005
                  tsmnet_causal_le_lag4_mean2  0.526640     0.551157    0.554639 0.975611 0.584684 -10.059343
                        latent_mean_p10_seed1  0.522651     0.555625    0.462292 1.049262 0.608666 -10.458316
                  tsmnet_causal_le_lag4_seed0  0.522479     0.540336    0.545559 1.002143 0.596751 -10.475463
                  tsmnet_causal_le_lag4_seed1  0.521899     0.544140    0.516574 1.041876 0.616268 -10.533426
                       tsmnet_causal_le_mean2  0.519047     0.554969    0.558846 0.985452 0.591482 -10.818711
                     latent_meanstd_p10_seed1  0.513967     0.548894    0.470815 1.037689 0.605599 -11.326655
                       tsmnet_causal_le_seed0  0.510196     0.549134    0.576447 1.026719 0.613068 -11.703755
                       tsmnet_causal_le_seed1  0.508010     0.535676    0.502601 1.034185 0.615276 -11.922360
                        latent_mean_p10_seed0  0.495551     0.518488    0.606612 1.137599 0.642881 -13.168231
                     latent_deepset_p40_mean2  0.485249     0.520888    0.503991 1.047168 0.625244 -14.198515
                     latent_deepset_p10_mean2  0.482538     0.516428    0.503901 1.049357 0.626875 -14.469549
                     latent_deepset_p40_seed1  0.481667     0.515534    0.465185 1.056347 0.629779 -14.556655
                     latent_deepset_p10_seed1  0.477781     0.510073    0.475602 1.058484 0.631801 -14.945300
