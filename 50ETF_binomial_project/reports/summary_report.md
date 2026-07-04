# 50ETF Option Binomial Pricing Project Report

Tree steps used in the run: 50

## Main outputs
- data/processed/510050_volatility.csv: 5-day, 30-day historical volatility and GARCH conditional volatilities.
- outputs/tables/option_pricing_results.csv: option rows with CRR/EQP prices under all volatility inputs.
- outputs/tables/error_summary.csv: aggregate error metrics.
- outputs/figures/: volatility, pricing, error and CRR-EQP comparison charts.

## Best model by RMSE in this run
- Model: eqp_garch_1_2_vol_price
- RMSE: 0.020631
- MAE: 0.012682
- Bias: -0.006675

## CRR vs EQP model difference
CRR fixes u=exp(sigma*sqrt(dt)), d=1/u and adjusts the risk-neutral probability p. EQP/Jarrow-Rudd fixes p=0.5 and embeds the risk-neutral drift into u and d.

| volatility_input   |     n |   mean_crr_minus_eqp |   mean_abs_difference |   max_abs_difference |
|:-------------------|------:|---------------------:|----------------------:|---------------------:|
| hist_vol_5d        | 56728 |          1.07274e-07 |           0.000115239 |           0.00443584 |
| hist_vol_30d       | 54704 |         -5.77716e-07 |           0.000110964 |           0.00394486 |
| garch_1_1_vol      | 57172 |         -2.03167e-06 |           9.35363e-05 |           0.00339121 |
| garch_1_2_vol      | 57172 |         -2.03133e-06 |           9.33984e-05 |           0.00333323 |
| garch_2_1_vol      | 57172 |         -2.06708e-06 |           9.35407e-05 |           0.0033902  |
| garch_2_2_vol      | 57172 |         -2.03021e-06 |           9.3739e-05  |           0.00337642 |

## Notes
The GARCH implementation is a self-contained Gaussian quasi-MLE implementation using scipy, so the project does not depend on the external arch package.
The baseline dividend yield q is set to 0.0 in config.py. If ETF dividend data are available, replace q with a dividend yield estimate or use dividend-adjusted underlying prices.