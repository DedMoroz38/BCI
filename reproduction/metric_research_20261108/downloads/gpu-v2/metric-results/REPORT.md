# Wave1: nested person-held-out CNNs

Complete protocol requires10held-out people and one later-session fold covering2people.
No old public-test tuning;all10public people enter the new CV.

           config  seed  loso_folds  later_folds  loso_mean_person_cell  later_cell  complete_protocol
  shallow16_class     0          10            1               0.473029    0.395791               True
  shallow16_class     1          10            1               0.516295    0.401613               True
shallow16_subject     0          10            1               0.493753    0.546000               True
shallow16_subject     1          10            1               0.521732    0.385594               True
  shallow40_class     0          10            1               0.508980    0.555934               True

Completed network fits:55. Partial seed sets are labeled explicitly.
Next:combine saved predictions with matching local all10 Lag4 controls;rank official pooled OOF cellBA and report person-mean separately.
