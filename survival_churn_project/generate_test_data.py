"""Generate a realistic, reproducible account-level survival-analysis dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_ACCOUNTS = 1_500
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_FILE = DATA_DIR / "account_churn_survival.csv"


def generate_accounts(n_accounts: int = N_ACCOUNTS, seed: int = SEED) -> pd.DataFrame:
    """Create accounts with noisy health signals and churn/censoring times."""
    rng = np.random.default_rng(seed)
    segments = rng.choice(["SMB", "Mid-Market", "Enterprise"], n_accounts, p=[0.48, 0.35, 0.17])
    plan_by_segment = {
        "SMB": ["Basic", "Pro"],
        "Mid-Market": ["Pro", "Enterprise"],
        "Enterprise": ["Enterprise", "Pro"],
    }
    plans = np.array([rng.choice(plan_by_segment[s], p=[0.58, 0.42]) for s in segments])
    renewal_type = rng.choice(["Monthly", "Annual", "Multi-year"], n_accounts, p=[0.30, 0.55, 0.15])
    seats = np.maximum(5, rng.lognormal(mean=3.15 + (segments == "Enterprise") * 1.0, sigma=0.7, size=n_accounts).round().astype(int))
    utilisation = np.clip(rng.beta(7, 3, n_accounts) - (renewal_type == "Monthly") * 0.04, 0.05, 0.99)
    active_users = np.maximum(1, np.round(seats * utilisation).astype(int))
    arr = np.round(seats * rng.uniform(65, 145, n_accounts) * np.where(plans == "Enterprise", 1.25, 1.0), -2)
    login_days = np.clip(np.round(rng.normal(15 + utilisation * 12, 6, n_accounts)), 0, 30).astype(int)
    usage_change = np.clip(rng.normal(3 - (1 - utilisation) * 24, 20, n_accounts), -80, 80).round(1)
    tickets = rng.poisson(1.5 + (1 - utilisation) * 5, n_accounts)
    days_since_csm = np.clip(rng.gamma(2.1, 24, n_accounts).round(), 1, 240).astype(int)
    nps = np.clip(np.round(rng.normal(35 + utilisation * 20 - tickets * 1.2, 20, n_accounts)), -100, 100).astype(int)
    months_to_renewal = np.clip(rng.uniform(0.2, 18, n_accounts), 0.1, 18).round(1)
    champion_left = rng.binomial(1, np.clip(0.04 + (1 - utilisation) * 0.10, 0.02, 0.25))

    # A proportional-hazards-style signal makes adverse health characteristics
    # more likely to produce earlier churn, while noise keeps the data realistic.
    log_hazard = (
        -3.25
        + 0.95 * (1 - utilisation)
        + 0.55 * (usage_change < -15)
        + 0.018 * tickets
        + 0.004 * days_since_csm
        - 0.010 * nps
        + 0.80 * champion_left
        + 0.55 * (renewal_type == "Monthly")
        + 0.12 * (months_to_renewal < 2)
        - 0.12 * (segments == "Enterprise")
        + rng.normal(0, 0.28, n_accounts)
    )
    monthly_hazard = np.exp(log_hazard)
    event_time = rng.exponential(1 / monthly_hazard)
    observation_end = rng.uniform(6, 30, n_accounts)
    churned = (event_time <= observation_end).astype(int)
    duration = np.minimum(event_time, observation_end)

    return pd.DataFrame(
        {
            "account_id": [f"ACC-{i:05d}" for i in range(1, n_accounts + 1)],
            "snapshot_date": pd.Timestamp("2025-01-01") - pd.to_timedelta(rng.integers(0, 365, n_accounts), unit="D"),
            "duration_months": np.round(np.maximum(duration, 0.1), 2),
            "churned": churned,
            "arr_usd": arr,
            "seats": seats,
            "active_users_30d": active_users,
            "seat_utilisation_pct": np.round(utilisation * 100, 1),
            "login_days_30d": login_days,
            "usage_change_90d_pct": usage_change,
            "support_tickets_90d": tickets,
            "days_since_csm_contact": days_since_csm,
            "nps": nps,
            "months_to_renewal": months_to_renewal,
            "champion_left_90d": champion_left,
            "segment": segments,
            "plan_type": plans,
            "renewal_type": renewal_type,
        }
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame = generate_accounts()
    frame.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d")
    print(f"Generated {len(frame):,} accounts at {OUTPUT_FILE}")
    print(f"Observed churn events: {frame['churned'].sum():,} ({frame['churned'].mean():.1%})")


if __name__ == "__main__":
    main()

