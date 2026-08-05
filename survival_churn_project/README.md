# Account survivorship curve

This project has one purpose: upload one account-history CSV and create one Kaplan–Meier survivorship curve.

## Start here

### 1. Export account data from Salesforce

Export one row per account from an Account, Subscription, or Customer Success report. Pull these fields:

| Salesforce field | Purpose |
|---|---|
| Account Name | Name shown in the file |
| Subscription Start Date (or Account Created Date) | When observation begins |
| Cancellation Date (or Churn Date) | When the account churned, if applicable |
| Account Status | Identifies active versus churned accounts |
| Report/Snapshot Date | End of observation for active accounts |

Before uploading, save a CSV with these final columns:

```csv
account_name,duration_months,churned
Acme Ltd,12,1
Bright Co,18,0
Northwind,24,0
```

- `duration_months` is the number of months from the start date to the cancellation date. For an active account, calculate it from the start date to the report/snapshot date.
- `churned` is `1` when the account cancelled and `0` when it was still active at the report/snapshot date.

You can export the Salesforce report, add the two final columns in Excel or Google Sheets, and save the file as `account_history.csv`.

An amount or risk score alone cannot create a true Kaplan–Meier curve. The chart needs both the time observed and the churn outcome.

### 2. Run the dashboard

Open Terminal and run:

```bash
cd /path/to/Churn-Survival/survival_churn_project
python3 -m pip install -r simple_requirements.txt
python3 -m streamlit run simple_dashboard.py
```

Replace `/path/to/Churn-Survival` with the location where the project is saved. A browser window will open. Upload `account_history.csv`; the single survivorship curve will appear.

## What the chart means

The curve estimates the probability that an account remains active as time passes.

- The horizontal axis shows months since observation began.
- The vertical axis shows the estimated probability of remaining active.
- A downward step represents observed churn events.
- Accounts still active at the end of the observation period are included as `churned = 0`. This means “still active when we stopped observing,” not “will never churn.”

The dashboard also shows a confidence band and an estimated median survival time when the data supports it. It produces no scatter plots, risk-band charts, category comparisons, or churn-prediction files.

