# RaceCast Research — Empirical Results

## Dataset
- Segment observations: **2,487**
- Same-driver lap comparison pairs: **372**
- Circuits: **5**
- Drivers represented in segment dataset: **21**

### Per-circuit processed dataset
| event       |   selected_laps |   drivers_selected |   comparison_pairs |
|:------------|----------------:|-------------------:|-------------------:|
| Bahrain     |              85 |                 20 |                 65 |
| Monaco      |             143 |                 20 |                122 |
| Monza       |              87 |                 20 |                 67 |
| Silverstone |              80 |                 15 |                 65 |
| Suzuka      |              74 |                 20 |                 53 |

## Cross-circuit model comparison
| model             |   MAE_mean |   MAE_std |   RMSE_mean |   RMSE_std |   R2_mean |   R2_std |
|:------------------|-----------:|----------:|------------:|-----------:|----------:|---------:|
| Gradient Boosting |   0.056346 |  0.031031 |    0.093504 |   0.048633 |  0.725774 | 0.038905 |
| Random Forest     |   0.057509 |  0.033515 |    0.101162 |   0.059318 |  0.696534 | 0.057372 |
| Linear Regression |   0.070060 |  0.037269 |    0.104454 |   0.055746 |  0.626063 | 0.187187 |

Best mean held-out-circuit MAE: **Gradient Boosting = 0.056346 s**.

### Leave-one-circuit-out results
| model             | held_out_circuit   |   n_test |      MAE |     RMSE |       R2 |
|:------------------|:-------------------|---------:|---------:|---------:|---------:|
| Linear Regression | Bahrain            |      520 | 0.042943 | 0.057625 | 0.637124 |
| Linear Regression | Monaco             |      854 | 0.095431 | 0.140077 | 0.325516 |
| Linear Regression | Monza              |      402 | 0.050855 | 0.092198 | 0.839286 |
| Linear Regression | Silverstone        |      385 | 0.122895 | 0.181542 | 0.638378 |
| Linear Regression | Suzuka             |      326 | 0.038175 | 0.050828 | 0.690010 |
| Random Forest     | Bahrain            |      520 | 0.032575 | 0.049854 | 0.728392 |
| Random Forest     | Monaco             |      854 | 0.060493 | 0.096065 | 0.682775 |
| Random Forest     | Monza              |      402 | 0.047964 | 0.129637 | 0.682262 |
| Random Forest     | Silverstone        |      385 | 0.113725 | 0.186595 | 0.617969 |
| Random Forest     | Suzuka             |      326 | 0.032791 | 0.043660 | 0.771271 |
| Gradient Boosting | Bahrain            |      520 | 0.034395 | 0.051232 | 0.713175 |
| Gradient Boosting | Monaco             |      854 | 0.061949 | 0.097007 | 0.676528 |
| Gradient Boosting | Monza              |      402 | 0.044878 | 0.112504 | 0.760700 |
| Gradient Boosting | Silverstone        |      385 | 0.107804 | 0.162991 | 0.708507 |
| Gradient Boosting | Suzuka             |      326 | 0.032704 | 0.043785 | 0.769959 |

## Strongest Spearman associations
| feature                 |    n |   spearman_rho |   p_value |
|:------------------------|-----:|---------------:|----------:|
| mean_speed_delta_kph    | 2487 |      -0.823417 |  0.000000 |
| gear_mean_delta         | 2487 |      -0.448696 |  0.000000 |
| min_speed_delta_kph     | 2487 |      -0.448112 |  0.000000 |
| throttle_mean_delta_pct | 2487 |      -0.369052 |  0.000000 |
| rpm_mean_delta          | 2487 |      -0.328723 |  0.000000 |

## Feature-group ablation
| configuration             |   MAE_mean |   MAE_std |   delta_MAE_vs_full |
|:--------------------------|-----------:|----------:|--------------------:|
| Full model                |   0.056346 |  0.031031 |            0.000000 |
| Without braking           |   0.055994 |  0.030400 |           -0.000352 |
| Without corner_speed      |   0.083634 |  0.033422 |            0.027288 |
| Without throttle          |   0.055804 |  0.028487 |           -0.000542 |
| Without exit_acceleration |   0.056756 |  0.032345 |            0.000410 |
| Without powertrain        |   0.053535 |  0.023636 |           -0.002811 |

## Model-agnostic feature importance
| feature                  |   importance_mean |   importance_std |
|:-------------------------|------------------:|-----------------:|
| mean_speed_delta_kph     |          0.140801 |         0.001268 |
| reference_segment_time_s |          0.056005 |         0.000659 |
| min_speed_delta_kph      |          0.024199 |         0.000569 |
| segment_length_m         |          0.022168 |         0.000319 |
| rpm_mean_delta           |          0.007136 |         0.000239 |

## SHAP
SHAP generated: **True**

## Methodological caution
These values describe predictive associations in public telemetry. They do not establish physical causality or reproduce proprietary team telemetry.
