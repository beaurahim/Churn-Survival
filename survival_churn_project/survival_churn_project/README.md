# Account churn survival analysis

This is a complete synthetic example for RevOps teams that want to estimate how long customer accounts remain active and identify characteristics associated with earlier churn.

## Install and run

From this directory:

```bash
python -m pip install -r requirements.txt
python generate_test_data.py
python train_survival_model.py
```

The scripts create `data/` and `outputs/` automatically. The generated CSV is `data/account_churn_survival.csv`.

## Simplest option for non-technical stakeholders

If stakeholders only need a simple account list, amount, and risk score, use the upload dashboard instead of the survival-analysis workflow:

```bash
python3 -m pip install -r simple_requirements.txt
streamlit run simple_dashboard.py
```

This opens a web page in the browser. Upload a CSV such as:

```text
account_name,amount,risk_score
Acme Ltd,125000,0.62
Bright Co,48000,0.18
Northwind,210000,0.76
```

The dashboard accepts risk scores written either as decimals (`0.62`) or percentages (`62`). It produces summary cards, a count of accounts by Low/Medium/High/Critical risk, an amount-at-risk chart, and a sortable account table. It also lets users download the cleaned risk list. The column names do not have to match the example because the dashboard asks the user to select the account, amount, and risk columns.

## What the model means

A normal churn classifier asks: “Will this account churn within a fixed window?” A survival model asks: “How does this account’s probability of remaining active change as time passes?”

The Kaplan–Meier survival curve starts near 1 because almost all accounts are active at month zero. It declines as observed churn events occur. It does not need to reach zero: some accounts remain active when the observation window ends.

`duration_months` is the time-to-event variable. `churned = 1` means churn was observed. `churned = 0` means the observation is right-censored: the account was still active at the end of observation. An active account is not assumed never to churn; we only know it survived until the observation ended. The Kaplan–Meier curve estimates the probability of remaining active beyond each month.

## Outputs

- `outputs/account_survivorship_curve.png`: overall Kaplan–Meier curve with confidence interval and 50% reference line.
- `outputs/km_seat_utilisation.png`, `km_champion_left.png`, `km_renewal_type.png`, `km_segment.png`: category comparisons.
- `outputs/cox_model_coefficients.csv` and `.png`: regularised Cox effects, hazard ratios, p-values, and confidence intervals.
- `outputs/account_churn_predictions.csv`: held-out account predictions sorted by 3-month churn probability.
- `outputs/calibration_3_6_12_months.csv`: descriptive horizon calibration checks.

The account prediction file contains churn probabilities within 1, 3, 6, and 12 months, predicted median months to churn when estimable, relative risk, and configurable Low/Medium/High/Critical risk bands. Default 3-month thresholds are below 10%, 10–25%, 25–50%, and 50% or above.

## Regularisation and evaluation

The Cox model uses `lifelines.CoxPHFitter` with Elastic Net shrinkage. `l1_ratio = 0` is Ridge, `l1_ratio = 1` is Lasso, and a value between 0 and 1 is Elastic Net. Shrinkage reduces overfitting and limits unstable coefficients, which is particularly useful when account-health variables are correlated. The script searches four `penalizer` values and four `l1_ratio` values, choosing the highest validation concordance index.

Preprocessing is fitted only on the training split. It drops identifiers and snapshot dates, imputes missing values, standardises numeric features, and one-hot encodes categories. The concordance index evaluates ranking: whether higher-risk accounts tend to churn earlier. It does not by itself prove that predicted probabilities are calibrated. Calibration output is descriptive because censoring requires more specialised estimators for a production-grade calibration study.

The proportional-hazards diagnostic is advisory. If lifelines reports a variable that may violate the assumption, review it and consider time-varying effects or stratification. Diagnostic warnings do not crash the project.

Association does not prove causation. A positive coefficient for a CSM-related variable does not prove that a CSM action caused or prevented churn.

## Replacing the synthetic CSV with real data

Replace `data/account_churn_survival.csv` with one row per account at a clearly defined historical snapshot. Join CRM/account attributes, product-usage telemetry, support history, and customer-success activity before running the model. Keep the column names and meanings in the provided schema, or update the validation and feature lists together.

Strong leakage warning: every predictor must have been known at the snapshot date. Do not use cancellation requests, churn reasons, final invoices, account closure fields, or any information recorded after the prediction date. Define an observation end date, mark accounts that churned during the window as `churned = 1`, and mark accounts still active at the end as `churned = 0` with their observed duration.

