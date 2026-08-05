"""Simple upload-and-visualise Kaplan–Meier dashboard."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lifelines import KaplanMeierFitter


st.set_page_config(page_title="Account Survivor Curve", page_icon="📈", layout="centered")


def choose_column(columns: list[str], likely_names: list[str]) -> str:
    lowered = {name.lower(): name for name in columns}
    for likely in likely_names:
        if likely in lowered:
            return lowered[likely]
    return columns[0]


def main() -> None:
    st.title("Account Survivor Curve")
    st.write(
        "Upload account history and this page will create one Kaplan–Meier chart showing "
        "the estimated share of accounts still active over time."
    )

    with st.expander("What should be in the CSV?", expanded=True):
        st.write("Your file needs these two columns:")
        st.code(
            "duration_months,churned\n"
            "6,1\n"
            "12,0\n"
            "18,1\n"
            "24,0",
            language="text",
        )
        st.caption(
            "duration_months = how many months the account was observed. "
            "churned = 1 if it churned during that period, and 0 if it was still active "
            "when observation ended. An account name column is optional."
        )

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded is None:
        st.info("Choose a CSV above to see the survivor curve.")
        st.stop()

    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"I could not read this CSV: {exc}")
        st.stop()
    if frame.empty:
        st.error("This CSV is empty.")
        st.stop()

    columns = frame.columns.tolist()
    numeric = frame.select_dtypes(include="number").columns.tolist()
    if not numeric:
        st.error("I could not find numeric columns. duration_months and churned must be numbers.")
        st.stop()

    st.sidebar.header("Choose the columns")
    duration_col = st.sidebar.selectbox(
        "Months observed", numeric, index=numeric.index(choose_column(numeric, ["duration_months", "duration", "months"])),
    )
    event_options = [column for column in numeric if column != duration_col] or numeric
    churn_col = st.sidebar.selectbox(
        "Churned? (1 = yes, 0 = still active)",
        event_options,
        index=event_options.index(choose_column(event_options, ["churned", "event", "churn"])),
    )

    duration = pd.to_numeric(frame[duration_col], errors="coerce")
    churned = pd.to_numeric(frame[churn_col], errors="coerce")
    clean = pd.DataFrame({"duration_months": duration, "churned": churned}).dropna()
    clean = clean[clean["duration_months"] > 0]
    if not clean["churned"].isin([0, 1]).all():
        st.error("The churned column must contain only 0 (still active) or 1 (churned).")
        st.stop()
    if clean.empty:
        st.error("There are no usable rows. Check duration_months and churned.")
        st.stop()

    kmf = KaplanMeierFitter()
    kmf.fit(clean["duration_months"], event_observed=clean["churned"], label="All accounts")

    st.subheader("Estimated account survivorship")
    st.caption(
        "The curve shows the estimated probability that an account remains active beyond each month. "
        "The shaded area is the uncertainty range."
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#1769aa")
    ax.axhline(0.5, color="#b04a4a", linestyle="--", linewidth=1, label="50% survival")
    median = kmf.median_survival_time_
    if np.isfinite(median):
        ax.axvline(median, color="#b04a4a", linestyle=":", linewidth=1)
        median_text = f"Estimated median survival: {median:.1f} months"
    else:
        median_text = "Median survival was not reached during observation"
    ax.set_title(f"Kaplan–Meier Survivor Curve\n{median_text}")
    ax.set_xlabel("Months since observation began")
    ax.set_ylabel("Probability account remains active")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        f"Based on {len(clean):,} accounts and {int(clean['churned'].sum()):,} observed churn events."
    )


if __name__ == "__main__":
    main()

