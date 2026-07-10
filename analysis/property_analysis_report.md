# Merged Property Analysis Report

## Global Summary

- Analyzed properties: 50
- Total data points across analyzed properties: 1913874
- Sum of property-level unique systems: 1617417
- Properties recommended for system holdout test: 43
- Properties recommended for grouped CV: 7
- Properties recommended for leave-one-system-out or descriptive analysis: 0

## Largest Properties

| bucket | property | data_points | unique_systems | recommended_split |
| --- | --- | --- | --- | --- |
| simulation | solvation | 999968 | 999968 | system_holdout_test |
| experiment | density | 98517 | 6206 | system_holdout_test |
| experiment | viscosity | 45804 | 2750 | system_holdout_test |
| experiment | solvation | 44400 | 3611 | system_holdout_test |
| simulation | esp_min | 28213 | 27275 | system_holdout_test |
| simulation | gap | 28213 | 27275 | system_holdout_test |
| simulation | esp_std | 28213 | 27275 | system_holdout_test |
| simulation | esp_abs_mean | 28213 | 27275 | system_holdout_test |
| simulation | esp_max | 28213 | 27275 | system_holdout_test |
| simulation | q_max | 28212 | 27274 | system_holdout_test |
| simulation | q_std | 28212 | 27274 | system_holdout_test |
| simulation | q_abs_mean | 28212 | 27274 | system_holdout_test |

## High Leakage Risk Properties

These properties should not be split by random rows because repeated systems can cross train/test boundaries.

| bucket | property | unique_systems | multi_point_system_ratio | max_points_per_system |
| --- | --- | --- | --- | --- |
| experiment | dynamic_relative_permittivity | 49 | 1.0 | 18 |
| simulation | heat_of_vaporization | 13441 | 0.9965032363663417 | 2 |
| simulation | density | 12554 | 0.9949816791460889 | 9 |
| experiment | x_co2 | 124 | 0.9919354838709677 | 671 |
| experiment | equilibrium_pressure | 95 | 0.9894736842105263 | 151 |
| experiment | speed_of_sound | 216 | 0.9861111111111112 | 252 |
| experiment | thermal_conductivity | 112 | 0.9821428571428571 | 75 |
| experiment | heat_capacity | 430 | 0.9441860465116279 | 1732 |
| experiment | surface_tension | 1255 | 0.8302788844621514 | 163 |
| experiment | electrical_conductivity | 705 | 0.8042553191489362 | 319 |
| experiment | density | 6206 | 0.7982597486303578 | 1844 |
| experiment | refractive_index | 754 | 0.7970822281167109 | 237 |

## System Holdout Test Candidates

| bucket | property | unique_systems | recommended_test_systems |
| --- | --- | --- | --- |
| simulation | solvation | 999968 | 99997 |
| experiment | density | 6206 | 621 |
| experiment | viscosity | 2750 | 275 |
| experiment | solvation | 3611 | 362 |
| simulation | gap | 27275 | 2728 |
| simulation | esp_min | 27275 | 2728 |
| simulation | esp_max | 27275 | 2728 |
| simulation | esp_std | 27275 | 2728 |
| simulation | esp_abs_mean | 27275 | 2728 |
| simulation | q_max | 27274 | 2728 |
| simulation | q_neg_sum | 27274 | 2728 |
| simulation | q_min | 27274 | 2728 |

## Grouped Cross-Validation Candidates

| bucket | property | unique_systems | data_points |
| --- | --- | --- | --- |
| experiment | isobaric_coefficient_of_volume_expansion | 25 | 894 |
| experiment | self_diffusion_coefficient | 36 | 382 |
| experiment | static_relative_permittivity | 44 | 65 |
| experiment | dynamic_relative_permittivity | 49 | 882 |
| experiment | equilibrium_pressure | 95 | 2029 |
| experiment | thermal_conductivity | 112 | 1597 |
| experiment | x_co2 | 124 | 9683 |

## Small Properties

None.

## Generated Figures

- condition_availability: 1
- coverage: 4
- normalized_property_violin: 1
- property_distribution_1d: 50
- property_distribution_2d: 53
- system_frequency: 50

Key summary figures:
- `figures/coverage/property_coverage_all.png`
- `figures/condition_availability/property_condition_availability_heatmap.png`
- `figures/normalized_distributions/property_normalized_violin.png`