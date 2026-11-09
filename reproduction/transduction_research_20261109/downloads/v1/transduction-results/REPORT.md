# Causal transduction research

11/11 outer folds complete; 123 configurations x 2 saved priors.
Main metric: pooled OOF mean subject/session/context balanced accuracy. Later-session metric is separate.
Exploratory public-CV comparison; all probabilities emitted from current/past inputs only.

                           config     prior  oof_cell  mean_person  later_cell      nll    brier  delta_pp
         lag_running__soft50_half lag4_swad  0.627234     0.649228    0.711102 0.862356 0.508482  2.109522
           lag_log__soft50_ramp60 lag4_swad  0.624553     0.645936    0.700527 0.923356 0.529080  1.841411
                  views_stable005 lag4_swad  0.624147     0.642454    0.719167 0.916961 0.524435  1.800807
        lag_pseudo__soft50_ramp60 lag4_swad  0.624036     0.642426    0.705978 0.911735 0.525227  1.789765
              lag_running__soft75 lag4_swad  0.624032     0.643846    0.697992 0.955187 0.538509  1.789313
                  views_mean_half lag4_swad  0.623771     0.643329    0.708819 0.860975 0.507608  1.763276
     lag_running20__soft50_ramp60 lag4_swad  0.623059     0.644075    0.697818 0.915399 0.526934  1.692021
     lag_running__soft50_shrink08 lag4_swad  0.622994     0.643708    0.695630 0.888787 0.520221  1.685575
  lag_pseudo_sharp__soft50_ramp60 lag4_swad  0.622931     0.641562    0.708666 0.914872 0.527199  1.679278
     joint_running__soft50_ramp60 lag4_swad  0.622708     0.641090    0.695604 0.904581 0.522340  1.656927
       joint_running__soft50_half lag4_swad  0.622692     0.643198    0.703501 0.861866 0.508071  1.655318
               lag_pseudo__soft25 lag4_swad  0.622603     0.641374    0.697997 0.973170 0.540821  1.646482
            lag_running20__soft75 lag4_swad  0.622551     0.643108    0.689985 0.965839 0.542831  1.641252
              lag_running__soft25 lag4_swad  0.622240     0.640487    0.697997 0.971475 0.540131  1.610188
       lag_running20__no_marginal lag4_swad  0.622238     0.642268    0.697517 0.993721 0.544502  1.609988
       lag_running__soft50_ramp60 lag4_swad  0.622163     0.640856    0.703374 0.910707 0.524583  1.602473
ordinary_running20__soft50_ramp60 lag4_swad  0.621629     0.644439    0.685475 0.916955 0.529854  1.549033
                  lag_log__soft50 lag4_swad  0.621515     0.644171    0.700750 0.965144 0.543820  1.537631
          lag_pseudo__no_marginal lag4_swad  0.621510     0.641581    0.697844 0.987976 0.542389  1.537155
     joint_running__soft50_beta05 lag4_swad  0.621286     0.642019    0.713400 0.888406 0.527571  1.514702
                   views_stable02 lag4_swad  0.621190     0.642105    0.703764 0.931114 0.529024  1.505177
   joint_running__soft50_shrink08 lag4_swad  0.621180     0.639265    0.698669 0.885291 0.518776  1.504188
            joint_running__soft75 lag4_swad  0.621163     0.640716    0.698209 0.952152 0.538249  1.502488
                lag_running__hard lag4_swad  0.621150     0.642894    0.682520 0.954336 0.539848  1.501151
         lag_running__no_marginal lag4_swad  0.621100     0.640302    0.695156 0.986935 0.541719  1.496192
        joint_running__soft50_k30 lag4_swad  0.621082     0.638888    0.700507 0.984998 0.547685  1.494351
               lag_pseudo__soft75 lag4_swad  0.621067     0.641061    0.697992 0.958056 0.539618  1.492814
     joint_running__soft50_warm50 lag4_swad  0.620922     0.638220    0.703590 0.939726 0.535598  1.478355
                 lag_pseudo__hard lag4_swad  0.620766     0.642356    0.685124 0.957566 0.541110  1.462748
            lag_running20__soft25 lag4_swad  0.620749     0.641203    0.694982 0.979278 0.543031  1.461086
