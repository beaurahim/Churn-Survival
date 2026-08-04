"""Train and evaluate a leakage-safe, regularised Cox churn-survival model."""

from pathlib import Path
from typing import Iterable
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.utils import concordance_index
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "account_churn_survival.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
HORIZONS = [1, 3, 6, 12]
RISK_THRESHOLDS = {"medium": 0.10, "high": 0.25, "critical": 0.50}
PENALIZERS = [0.001, 0.01, 0.1, 1.0]
L1_RATIOS = [0.0, 0.5, 0.9, 1.0]

REQUIRED_COLUMNS = {
    "account_id", "snapshot_date", "duration_months", "churned", "arr_usd", "seats",
    "active_users_30d", "seat_utilisation_pct", "login_days_30d", "usage_change_90d_pct",
    "support_tickets_90d", "days_since_csm_contact", "nps", "months_to_renewal",
    "champion_left_90d", "segment", "plan_type", "renewal_type",
}


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load the CSV and fail early on structural or survival-data problems."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run generate_test_data.py first.")
    df = pd.read_csv(path, parse_dates=["snapshot_date"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["account_id"].duplicated().any():
        raise ValueError("account_id must be unique.")
    if len(df) < 1_000:
        raise ValueError("The dataset must contain at least 1,000 accounts.")
    if not df["churned"].isin([0, 1]).all() or (df["duration_months"] <= 0).any():
        raise ValueError("churned must be 0/1 and duration_months must be positive.")
    if not df["snapshot_date"].notna().all():
        raise ValueError("snapshot_date contains invalid dates.")
    print(f"Validated {len(df):,} accounts and {int(df['churned'].sum()):,} observed churn events.")
    return df


def style_axes(ax: plt.Axes) -> None:
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def plot_overall_km(df: pd.DataFrame) -> None:
    kmf = KaplanMeierFitter()
    kmf.fit(df["duration_months"], event_observed=df["churned"], label="All accounts")
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#1769aa")
    ax.axhline(0.5, color="#b04a4a", linestyle="--", linewidth=1, label="50% survival")
    median = kmf.median_survival_time_
    subtitle = f"Estimated median survival: {median:.1f} months" if np.isfinite(median) else "Median survival not reached during observation"
    ax.set(title=f"Account Survivorship Curve\n{subtitle}", xlabel="Months since account snapshot", ylabel="Probability account remains active")
    style_axes(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "account_survivorship_curve.png", dpi=160)
    plt.close(fig)


def plot_grouped_km(df: pd.DataFrame, column: str, groups: Iterable[str], filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("colorblind", len(list(groups)))
    for color, group in zip(palette, groups):
        subset = df[df[column] == group]
        if subset.empty:
            continue
        KaplanMeierFitter().fit(subset["duration_months"], subset["churned"], label=f"{group} (n={len(subset):,})").plot_survival_function(ax=ax, ci_show=True, color=color)
    ax.set(title=title, xlabel="Months since account snapshot", ylabel="Probability account remains active")
    style_axes(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close(fig)


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Fit imputers/scalers/encoders later on training data only."""
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)], remainder="drop")


def transformed_frame(preprocessor: ColumnTransformer, raw: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    values = preprocessor.transform(raw)
    return pd.DataFrame(values, columns=feature_names, index=raw.index)


def fit_candidate(train_x: pd.DataFrame, train_duration: pd.Series, train_event: pd.Series, penalizer: float, l1_ratio: float) -> CoxPHFitter | None:
    design = train_x.copy()
    design["duration_months"] = train_duration
    design["churned"] = train_event
    model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(design, duration_col="duration_months", event_col="churned", show_progress=False)
        return model
    except (ValueError, np.linalg.LinAlgError, ConvergenceWarning) as exc:
        print(f"Skipped penalizer={penalizer}, l1_ratio={l1_ratio}: {exc}")
        return None


def c_index(model: CoxPHFitter, x: pd.DataFrame, duration: pd.Series, event: pd.Series) -> float:
    risk = model.predict_partial_hazard(x).to_numpy().ravel()
    return float(concordance_index(duration, -risk, event))


def save_coefficients(model: CoxPHFitter) -> None:
    summary = model.summary.reset_index().rename(columns={"covariate": "variable_name", "coef": "coefficient", "p": "p_value", "coef lower 95%": "confidence_interval_lower", "coef upper 95%": "confidence_interval_upper"})
    summary["hazard_ratio"] = np.exp(summary["coefficient"])
    summary["absolute_coefficient"] = summary["coefficient"].abs()
    columns = ["variable_name", "coefficient", "hazard_ratio", "absolute_coefficient", "p_value", "confidence_interval_lower", "confidence_interval_upper"]
    summary[columns].sort_values("absolute_coefficient", ascending=False).to_csv(OUTPUT_DIR / "cox_model_coefficients.csv", index=False)
    chart = summary.sort_values("coefficient").tail(16)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#c44e52" if value > 0 else "#4c72b0" for value in chart["coefficient"]]
    ax.barh(chart["variable_name"], chart["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Largest Cox Model Effects", xlabel="Coefficient (positive = higher churn hazard)", ylabel="Variable")
    ax.grid(axis="x", alpha=0.25)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cox_model_coefficients.png", dpi=160)
    plt.close(fig)


def add_predictions(model: CoxPHFitter, test_x: pd.DataFrame, test_raw: pd.DataFrame) -> pd.DataFrame:
    survival = model.predict_survival_function(test_x, times=HORIZONS)
    predictions = test_raw[["account_id"]].copy()
    for horizon in HORIZONS:
        predictions[f"churn_probability_{horizon}m"] = 1 - survival.loc[horizon].to_numpy()
    predictions["relative_risk_score"] = model.predict_partial_hazard(test_x).to_numpy().ravel()
    predictions["predicted_median_months_to_churn"] = model.predict_median(test_x).to_numpy()
    p3 = predictions["churn_probability_3m"]
    predictions["risk_band"] = np.select([p3 >= RISK_THRESHOLDS["critical"], p3 >= RISK_THRESHOLDS["high"], p3 >= RISK_THRESHOLDS["medium"]], ["Critical", "High", "Medium"], default="Low")
    return predictions.sort_values("churn_probability_3m", ascending=False)


def calibration_table(model: CoxPHFitter, x: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    survival = model.predict_survival_function(x, times=HORIZONS)
    rows = []
    for horizon in [3, 6, 12]:
        known = (raw["duration_months"] >= horizon) | (raw["churned"] == 1)
        observed = ((raw["churned"] == 1) & (raw["duration_months"] <= horizon))[known]
        predicted = (1 - survival.loc[horizon])[known.to_numpy()]
        rows.append({"horizon_months": horizon, "accounts_with_known_outcome": int(known.sum()), "mean_predicted_churn_probability": float(np.mean(predicted)), "observed_churn_rate_among_known": float(observed.mean())})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_validate(DATA_FILE)
    plot_overall_km(df)
    df["utilisation_band"] = np.where(df["seat_utilisation_pct"] >= df["seat_utilisation_pct"].median(), "High utilisation", "Low utilisation")
    plot_grouped_km(df, "utilisation_band", ["High utilisation", "Low utilisation"], "km_seat_utilisation.png", "Survivorship by Seat Utilisation")
    plot_grouped_km(df, "champion_left_90d", [0, 1], "km_champion_left.png", "Survivorship by Champion Status")
    plot_grouped_km(df, "renewal_type", ["Monthly", "Annual", "Multi-year"], "km_renewal_type.png", "Survivorship by Renewal Type")
    plot_grouped_km(df, "segment", ["SMB", "Mid-Market", "Enterprise"], "km_segment.png", "Survivorship by Segment")

    train_val, test = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["churned"])
    train, validation = train_test_split(train_val, test_size=0.25, random_state=SEED, stratify=train_val["churned"])
    target = ["account_id", "snapshot_date", "duration_months", "churned", "utilisation_band"]
    predictors = [c for c in df.columns if c not in target]
    categorical = ["segment", "plan_type", "renewal_type"]
    numeric = [c for c in predictors if c not in categorical]
    preprocessor = make_preprocessor(numeric, categorical)
    train_x = pd.DataFrame(preprocessor.fit_transform(train[predictors]), columns=preprocessor.get_feature_names_out(), index=train.index)
    val_x = transformed_frame(preprocessor, validation[predictors], list(train_x.columns))
    test_x = transformed_frame(preprocessor, test[predictors], list(train_x.columns))

    best_model, best_score, best_params = None, -np.inf, None
    for penalizer in PENALIZERS:
        for l1_ratio in L1_RATIOS:
            model = fit_candidate(train_x, train["duration_months"], train["churned"], penalizer, l1_ratio)
            if model is None:
                continue
            score = c_index(model, val_x, validation["duration_months"], validation["churned"])
            print(f"penalizer={penalizer:<5} l1_ratio={l1_ratio:<3} validation_c_index={score:.3f}")
            if score > best_score:
                best_model, best_score, best_params = model, score, (penalizer, l1_ratio)
    if best_model is None:
        raise RuntimeError("No Cox model converged. Check the input data or reduce regularisation.")

    # The diagnostic is advisory: small synthetic datasets can trigger warnings.
    try:
        with warnings.catch_warnings(record=True) as diagnostic_warnings:
            warnings.simplefilter("always")
            best_model.check_assumptions(pd.concat([train_x, train[["duration_months", "churned"]]], axis=1), p_value_threshold=0.05, show_plots=False)
            if diagnostic_warnings:
                print("PH diagnostic warning: review the printed lifelines output; the warning did not stop training.")
    except Exception as exc:
        print(f"PH diagnostic could not complete (non-fatal): {exc}")

    save_coefficients(best_model)
    predictions = add_predictions(best_model, test_x, test)
    predictions.to_csv(OUTPUT_DIR / "account_churn_predictions.csv", index=False)
    calibration = calibration_table(best_model, test_x, test)
    calibration.to_csv(OUTPUT_DIR / "calibration_3_6_12_months.csv", index=False)

    train_c = c_index(best_model, train_x, train["duration_months"], train["churned"])
    test_c = c_index(best_model, test_x, test["duration_months"], test["churned"])
    print("\nEvaluation summary")
    print(f"Accounts: {len(df):,}; observed churn: {int(df['churned'].sum()):,} ({df['churned'].mean():.1%}); censoring: {(1 - df['churned'].mean()):.1%}")
    print(f"Training concordance index: {train_c:.3f}")
    print(f"Validation concordance index: {best_score:.3f}")
    print(f"Test concordance index: {test_c:.3f}")
    print(f"Best penalizer: {best_params[0]}; best l1_ratio: {best_params[1]}")
    print("\nCalibration (descriptive; censored-before-horizon accounts excluded):")
    print(calibration.to_string(index=False))
    print(f"\nSaved charts, coefficients, calibration, and predictions to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
