"""Simple upload-and-visualise dashboard for account risk CSV files."""

from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(page_title="Account Risk Dashboard", page_icon="📊", layout="wide")


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include="number").columns.tolist()


def choose_default(options: list[str], likely_names: list[str]) -> str:
    lowered = {name.lower(): name for name in options}
    for likely in likely_names:
        if likely in lowered:
            return lowered[likely]
    return options[0]


def normalise_risk(values: pd.Series) -> pd.Series:
    """Return risk as a 0–1 probability, accepting either 0–1 or 0–100 input."""
    numbers = pd.to_numeric(values, errors="coerce")
    if numbers.dropna().empty:
        return numbers
    if numbers.max() > 1:
        numbers = numbers / 100
    return numbers.clip(0, 1)


def risk_band(risk: pd.Series) -> pd.Series:
    return pd.cut(
        risk,
        bins=[-0.001, 0.10, 0.25, 0.50, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True,
    ).astype(str)


def format_money(value: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


def main() -> None:
    st.title("Account Risk Dashboard")
    st.write("Upload a CSV to see which accounts may need attention. No coding or survival-analysis knowledge is required.")

    uploaded = st.file_uploader("Upload your account CSV", type=["csv"])
    if uploaded is None:
        st.info("Your CSV needs one account-name column, one amount column, and one risk/probability column. Example: account_name, amount, risk_score")
        st.stop()

    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"I could not read this CSV: {exc}")
        st.stop()
    if frame.empty:
        st.error("This CSV is empty.")
        st.stop()

    number_options = numeric_columns(frame)
    if not number_options:
        st.error("I could not find any numeric columns for amount and risk.")
        st.stop()

    st.sidebar.header("Choose the columns")
    name_options = frame.columns.tolist()
    name_col = st.sidebar.selectbox("Account name column", name_options, index=name_options.index(choose_default(name_options, ["account_name", "account", "name", "account_id"])))
    amount_col = st.sidebar.selectbox("Amount column", number_options, index=number_options.index(choose_default(number_options, ["amount", "arr_usd", "arr", "revenue"])))
    risk_options = [column for column in number_options if column != amount_col] or number_options
    risk_col = st.sidebar.selectbox("Risk/probability column", risk_options, index=risk_options.index(choose_default(risk_options, ["risk_score", "risk", "probability", "churn_probability_3m"])))

    amount = pd.to_numeric(frame[amount_col], errors="coerce")
    risk = normalise_risk(frame[risk_col])
    clean = pd.DataFrame({"Account": frame[name_col].astype(str), "Amount": amount, "Risk probability": risk}).dropna()
    if clean.empty:
        st.error("The selected amount and risk columns do not contain usable numeric values.")
        st.stop()
    clean["Risk band"] = risk_band(clean["Risk probability"])
    clean = clean.sort_values("Risk probability", ascending=False)

    critical = clean["Risk band"].isin(["High", "Critical"])
    cards = st.columns(4)
    cards[0].metric("Accounts", f"{len(clean):,}")
    cards[1].metric("Total amount", format_money(clean["Amount"].sum()))
    cards[2].metric("Average risk", f"{clean['Risk probability'].mean():.1%}")
    cards[3].metric("High/Critical accounts", f"{critical.sum():,}")

    st.subheader("Risk overview")
    left, right = st.columns(2)
    order = ["Low", "Medium", "High", "Critical"]
    counts = clean["Risk band"].value_counts().reindex(order, fill_value=0)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=["#4c956c", "#e9c46a", "#f4a261", "#e76f51"], legend=False, ax=ax)
        ax.set(xlabel="Risk band", ylabel="Number of accounts", title="Accounts by risk band")
        ax.grid(axis="y", alpha=0.25)
        sns.despine(ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=clean, x="Risk probability", y="Amount", hue="Risk band", hue_order=order, palette={"Low": "#4c956c", "Medium": "#e9c46a", "High": "#f4a261", "Critical": "#e76f51"}, s=70, ax=ax)
        ax.set(xlabel="Churn/risk probability", ylabel="Amount", title="Amount at risk")
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        ax.yaxis.set_major_formatter(lambda value, _: format_money(value))
        ax.grid(alpha=0.2)
        sns.despine(ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Accounts needing attention")
    st.caption("Sorted from highest risk to lowest risk. Risk bands use Low <10%, Medium 10–25%, High 25–50%, Critical ≥50%.")
    st.dataframe(clean.style.format({"Amount": "${:,.0f}", "Risk probability": "{:.1%}"}), use_container_width=True, hide_index=True)

    download = BytesIO()
    clean.to_csv(download, index=False)
    st.download_button("Download this cleaned risk list", data=download.getvalue(), file_name="account_risk_summary.csv", mime="text/csv")


if __name__ == "__main__":
    main()
