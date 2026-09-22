# Merged Property Analysis Report

## Global Summary

- Analyzed properties: 39
- Total data points across analyzed properties: 1751483
- Sum of property-level unique systems: 1487433
- Properties recommended for system holdout test: 33
- Properties recommended for grouped CV: 6
- Properties recommended for leave-one-system-out or descriptive analysis: 0

## Largest Properties

| bucket | property | data_points | unique_systems | recommended_split |
| --- | --- | --- | --- | --- |
| simulation | transfer_organic | 1000000 | 1000000 | system_holdout_test |
| experiment | density | 100579 | 5966 | system_holdout_test |
| simulation | density | 52575 | 26204 | system_holdout_test |
| simulation | heat_of_vaporization | 41701 | 20895 | system_holdout_test |
| experiment | viscosity | 39941 | 2586 | system_holdout_test |
| simulation | charge | 28203 | 28203 | system_holdout_test |
| simulation | q_min | 27258 | 27258 | system_holdout_test |
| simulation | esp_min | 27258 | 27258 | system_holdout_test |
| simulation | quadrupole | 27258 | 27258 | system_holdout_test |
| simulation | dipole | 27258 | 27258 | system_holdout_test |
| simulation | esp_pos_frac | 27258 | 27258 | system_holdout_test |
| simulation | esp_max | 27258 | 27258 | system_holdout_test |

## High Leakage Risk Properties

These properties should not be split by random rows because repeated systems can cross train/test boundaries.

| bucket | property | unique_systems | multi_point_system_ratio | max_points_per_system |
| --- | --- | --- | --- | --- |
| experiment | dynamic_relative_permittivity | 49 | 1.0 | 18 |
| simulation | heat_of_vaporization | 20895 | 0.9957406078009093 | 2 |
| simulation | density | 26204 | 0.9941993588765075 | 9 |
| experiment | x_co2 | 122 | 0.9918032786885246 | 671 |
| experiment | equilibrium_pressure | 95 | 0.9894736842105263 | 151 |
| experiment | speed_of_sound | 216 | 0.9861111111111112 | 252 |
| experiment | thermal_conductivity | 93 | 0.978494623655914 | 95 |
| experiment | density | 5966 | 0.9446865571572243 | 2509 |
| experiment | heat_capacity | 352 | 0.875 | 1757 |
| experiment | surface_tension | 1141 | 0.845749342681858 | 180 |
| experiment | refractive_index | 726 | 0.803030303030303 | 237 |
| experiment | electrical_conductivity | 703 | 0.7510668563300142 | 304 |

## System Holdout Test Candidates

| bucket | property | unique_systems | recommended_test_systems |
| --- | --- | --- | --- |
| simulation | transfer_organic | 1000000 | 100000 |
| experiment | density | 5966 | 597 |
| simulation | density | 26204 | 2621 |
| simulation | heat_of_vaporization | 20895 | 2090 |
| experiment | viscosity | 2586 | 259 |
| simulation | charge | 28203 | 2821 |
| simulation | q_min | 27258 | 2726 |
| simulation | q_max | 27258 | 2726 |
| simulation | gap | 27258 | 2726 |
| simulation | esp_min | 27258 | 2726 |
| simulation | quadrupole | 27258 | 2726 |
| simulation | dipole | 27258 | 2726 |

## Grouped Cross-Validation Candidates

| bucket | property | unique_systems | data_points |
| --- | --- | --- | --- |
| experiment | self_diffusion_coefficient | 36 | 382 |
| experiment | static_relative_permittivity | 44 | 65 |
| experiment | dynamic_relative_permittivity | 49 | 882 |
| experiment | thermal_conductivity | 93 | 1540 |
| experiment | equilibrium_pressure | 95 | 2029 |
| experiment | x_co2 | 122 | 9536 |

## Small Properties

None.

## Cross-property IL System Overlap

Experimental ionic-liquid systems are keyed by `(cation, anion)`; measurement conditions are ignored.

- Full pairwise statistics: `property_system_overlap.csv`
- Properties with IL systems: 19

## Generated Figures

- condition_availability: 1
- coverage: 4
- normalized_property_violin: 1
- property_distribution_1d: 39
- property_distribution_2d: 48
- property_system_overlap: 1
- system_frequency: 39

Key summary figures:
- `figures/coverage/property_coverage_all.png`
- `figures/condition_availability/property_condition_availability_heatmap.png`
- `figures/normalized_distributions/property_normalized_violin.png`
- `figures/property_system_overlap/experiment_property_system_overlap_heatmap.png`