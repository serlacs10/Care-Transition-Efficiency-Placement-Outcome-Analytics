import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Care Transition Efficiency Analytics",
    layout="wide"
)

st.title("Care Transition Efficiency & Placement Outcome Analytics")
st.caption("UAC Program pipeline analytics: CBP custody to HHS care to sponsor placement")

uploaded_file = st.sidebar.file_uploader("Upload HHS UAC CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload the HHS UAC CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace("*", "", regex=False)
    .str.replace("\n", " ", regex=False)
    .str.replace("\r", " ", regex=False)
    .str.replace("  ", " ", regex=False)
)

column_mapping = {
    "Date": "date",
    "Children apprehended and placed in CBP custody": "apprehended",
    "Children in CBP custody": "cbp_custody",
    "Children transferred out of CBP custody": "transferred",
    "Children in HHS Care": "hhs_care",
    "Children discharged from HHS Care": "discharged"
}

df = df.rename(columns=column_mapping)

required_columns = [
    "date",
    "apprehended",
    "cbp_custody",
    "transferred",
    "hhs_care",
    "discharged"
]

missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.write("Detected columns:")
    st.write(df.columns.tolist())
    st.stop()

df = df[required_columns].copy()

df["date"] = pd.to_datetime(df["date"], errors="coerce")

for col in ["apprehended", "cbp_custody", "transferred", "hhs_care", "discharged"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace(["nan", "None", "", "NA", "N/A"], np.nan)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date"]).sort_values("date")

for col in ["apprehended", "cbp_custody", "transferred", "hhs_care", "discharged"]:
    df[col] = df[col].fillna(0)

# KPI calculations
df["transfer_efficiency_ratio"] = np.where(
    df["cbp_custody"] > 0,
    df["transferred"] / df["cbp_custody"],
    0
)

df["discharge_effectiveness_index"] = np.where(
    df["hhs_care"] > 0,
    df["discharged"] / df["hhs_care"],
    0
)

df["pipeline_throughput"] = np.where(
    df["apprehended"] > 0,
    df["discharged"] / df["apprehended"],
    0
)

df["net_hhs_backlog_change"] = df["transferred"] - df["discharged"]
df["net_cbp_pressure_change"] = df["apprehended"] - df["transferred"]
df["total_pipeline_load"] = df["cbp_custody"] + df["hhs_care"]

df["weekday"] = df["date"].dt.day_name()
df["month"] = df["date"].dt.to_period("M").astype(str)
df["is_weekend"] = df["date"].dt.weekday >= 5

df["transfer_rolling_7d"] = df["transferred"].rolling(7, min_periods=1).mean()
df["discharge_rolling_7d"] = df["discharged"].rolling(7, min_periods=1).mean()
df["backlog_rolling_7d"] = df["net_hhs_backlog_change"].rolling(7, min_periods=1).mean()

df["outcome_stability_score"] = (
    1 - df["discharge_effectiveness_index"].rolling(14, min_periods=3).std()
).clip(lower=0)

df["outcome_stability_score"] = df["outcome_stability_score"].fillna(1)

st.sidebar.header("Filters")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) != 2:
    st.warning("Please select both start and end dates.")
    st.stop()

start_date, end_date = date_range

filtered = df[
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date)
].copy()

if filtered.empty:
    st.warning("No data available for the selected date range.")
    st.stop()

metric_view = st.sidebar.multiselect(
    "Show ratio metrics",
    [
        "Transfer Efficiency Ratio",
        "Discharge Effectiveness Index",
        "Pipeline Throughput",
        "Outcome Stability Score"
    ],
    default=[
        "Transfer Efficiency Ratio",
        "Discharge Effectiveness Index",
        "Pipeline Throughput"
    ]
)

transfer_threshold = st.sidebar.slider(
    "Transfer efficiency alert threshold",
    0.0, 1.0, 0.20, 0.01
)

discharge_threshold = st.sidebar.slider(
    "Discharge effectiveness alert threshold",
    0.0, 1.0, 0.05, 0.01
)

backlog_threshold = st.sidebar.number_input(
    "Daily HHS backlog alert threshold",
    min_value=0,
    value=100
)

total_apprehended = filtered["apprehended"].sum()
total_transferred = filtered["transferred"].sum()
total_discharged = filtered["discharged"].sum()

overall_transfer_efficiency = (
    total_transferred / filtered["cbp_custody"].sum()
    if filtered["cbp_custody"].sum() > 0 else 0
)

overall_discharge_effectiveness = (
    total_discharged / filtered["hhs_care"].sum()
    if filtered["hhs_care"].sum() > 0 else 0
)

overall_pipeline_throughput = (
    total_discharged / total_apprehended
    if total_apprehended > 0 else 0
)

backlog_accumulation_rate = filtered["net_hhs_backlog_change"].mean()
outcome_stability = filtered["outcome_stability_score"].mean()

st.subheader("Executive KPI Summary")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric("Transfer Efficiency Ratio", f"{overall_transfer_efficiency:.2%}")
kpi2.metric("Discharge Effectiveness", f"{overall_discharge_effectiveness:.2%}")
kpi3.metric("Pipeline Throughput", f"{overall_pipeline_throughput:.2%}")
kpi4.metric("Backlog Accumulation Rate", f"{backlog_accumulation_rate:,.0f}")
kpi5.metric("Outcome Stability Score", f"{outcome_stability:.2f}")

st.subheader("Threshold-Based Alerts")

alert_col1, alert_col2, alert_col3 = st.columns(3)

low_transfer_days = filtered[
    filtered["transfer_efficiency_ratio"] < transfer_threshold
]

low_discharge_days = filtered[
    filtered["discharge_effectiveness_index"] < discharge_threshold
]

high_backlog_days = filtered[
    filtered["net_hhs_backlog_change"] > backlog_threshold
]

with alert_col1:
    if len(low_transfer_days) > 0:
        st.error(f"{len(low_transfer_days)} days below transfer efficiency threshold")
    else:
        st.success("Transfer efficiency is within threshold")

with alert_col2:
    if len(low_discharge_days) > 0:
        st.error(f"{len(low_discharge_days)} days below discharge threshold")
    else:
        st.success("Discharge effectiveness is within threshold")

with alert_col3:
    if len(high_backlog_days) > 0:
        st.warning(f"{len(high_backlog_days)} high backlog accumulation days")
    else:
        st.success("No severe backlog accumulation detected")

st.subheader("Care Pipeline Flow Visualization")

flow_df = filtered[
    ["date", "apprehended", "transferred", "discharged"]
].melt(
    id_vars="date",
    var_name="Pipeline Stage",
    value_name="Children"
)

flow_labels = {
    "apprehended": "Apprehended / CBP Intake",
    "transferred": "Transferred to HHS",
    "discharged": "Discharged to Sponsor"
}

flow_df["Pipeline Stage"] = flow_df["Pipeline Stage"].map(flow_labels)

fig_flow = px.line(
    flow_df,
    x="date",
    y="Children",
    color="Pipeline Stage",
    markers=True,
    title="Daily Care Pipeline Movement"
)

st.plotly_chart(fig_flow, use_container_width=True)

st.subheader("Transfer and Discharge Efficiency Panels")

ratio_map = {
    "Transfer Efficiency Ratio": "transfer_efficiency_ratio",
    "Discharge Effectiveness Index": "discharge_effectiveness_index",
    "Pipeline Throughput": "pipeline_throughput",
    "Outcome Stability Score": "outcome_stability_score"
}

selected_ratio_cols = [ratio_map[x] for x in metric_view]

if selected_ratio_cols:
    ratio_df = filtered[["date"] + selected_ratio_cols].melt(
        id_vars="date",
        var_name="Metric",
        value_name="Ratio"
    )

    ratio_names = {
        "transfer_efficiency_ratio": "Transfer Efficiency Ratio",
        "discharge_effectiveness_index": "Discharge Effectiveness Index",
        "pipeline_throughput": "Pipeline Throughput",
        "outcome_stability_score": "Outcome Stability Score"
    }

    ratio_df["Metric"] = ratio_df["Metric"].map(ratio_names)

    fig_ratio = px.line(
        ratio_df,
        x="date",
        y="Ratio",
        color="Metric",
        title="Efficiency and Outcome Ratios"
    )

    fig_ratio.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_ratio, use_container_width=True)

st.subheader("Bottleneck Detection Charts")

bottle_col1, bottle_col2 = st.columns(2)

with bottle_col1:
    fig_backlog = px.bar(
        filtered,
        x="date",
        y="net_hhs_backlog_change",
        title="HHS Backlog: Transfers Minus Discharges",
        color="net_hhs_backlog_change",
        color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(fig_backlog, use_container_width=True)

with bottle_col2:
    fig_cbp_pressure = px.bar(
        filtered,
        x="date",
        y="net_cbp_pressure_change",
        title="CBP Pressure: Apprehensions Minus Transfers",
        color="net_cbp_pressure_change",
        color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(fig_cbp_pressure, use_container_width=True)

st.subheader("Outcome Trend Analysis")

monthly = filtered.groupby("month", as_index=False).agg(
    apprehended=("apprehended", "sum"),
    transferred=("transferred", "sum"),
    discharged=("discharged", "sum"),
    avg_cbp_custody=("cbp_custody", "mean"),
    avg_hhs_care=("hhs_care", "mean"),
    backlog_change=("net_hhs_backlog_change", "sum")
)

fig_monthly = px.line(
    monthly,
    x="month",
    y=["apprehended", "transferred", "discharged"],
    markers=True,
    title="Month-over-Month Placement and Pipeline Trends"
)

st.plotly_chart(fig_monthly, use_container_width=True)

st.subheader("Weekday vs Weekend Transition Speed")

weekday_summary = filtered.groupby("is_weekend", as_index=False).agg(
    avg_transfers=("transferred", "mean"),
    avg_discharges=("discharged", "mean"),
    avg_transfer_efficiency=("transfer_efficiency_ratio", "mean"),
    avg_discharge_effectiveness=("discharge_effectiveness_index", "mean")
)

weekday_summary["Day Type"] = weekday_summary["is_weekend"].map({
    False: "Weekday",
    True: "Weekend"
})

fig_weekday = px.bar(
    weekday_summary,
    x="Day Type",
    y=["avg_transfers", "avg_discharges"],
    barmode="group",
    title="Average Transfers and Discharges: Weekday vs Weekend"
)

st.plotly_chart(fig_weekday, use_container_width=True)

st.subheader("Prolonged Stagnation Detection")

filtered["low_transfer_flag"] = filtered["transfer_efficiency_ratio"] < transfer_threshold
filtered["low_discharge_flag"] = filtered["discharge_effectiveness_index"] < discharge_threshold
filtered["high_backlog_flag"] = filtered["net_hhs_backlog_change"] > backlog_threshold

stagnation = filtered[
    filtered["low_transfer_flag"] |
    filtered["low_discharge_flag"] |
    filtered["high_backlog_flag"]
][[
    "date",
    "transfer_efficiency_ratio",
    "discharge_effectiveness_index",
    "net_hhs_backlog_change",
    "net_cbp_pressure_change"
]]

if stagnation.empty:
    st.success("No stagnation periods detected under the selected thresholds.")
else:
    st.dataframe(
        stagnation.style.format({
            "transfer_efficiency_ratio": "{:.2%}",
            "discharge_effectiveness_index": "{:.2%}",
            "net_hhs_backlog_change": "{:,.0f}",
            "net_cbp_pressure_change": "{:,.0f}"
        }),
        use_container_width=True
    )

st.subheader("Research Insights and Recommendations")

worst_transfer_day = filtered.loc[
    filtered["transfer_efficiency_ratio"].idxmin()
]

worst_discharge_day = filtered.loc[
    filtered["discharge_effectiveness_index"].idxmin()
]

highest_backlog_day = filtered.loc[
    filtered["net_hhs_backlog_change"].idxmax()
]

st.markdown(f"""
### Key Findings

- Total children apprehended during selected period: **{total_apprehended:,.0f}**
- Total children transferred from CBP custody: **{total_transferred:,.0f}**
- Total children discharged from HHS care: **{total_discharged:,.0f}**
- Lowest transfer efficiency occurred on **{worst_transfer_day['date'].date()}**
- Lowest discharge effectiveness occurred on **{worst_discharge_day['date'].date()}**
- Highest HHS backlog accumulation occurred on **{highest_backlog_day['date'].date()}**

### Recommendations

1. Monitor days when CBP transfers do not keep pace with apprehensions.
2. Create operational alerts when HHS discharges fall below expected levels.
3. Compare weekday and weekend performance to identify staffing gaps.
4. Use backlog trends to forecast shelter and case management pressure.
5. Track outcome stability to detect sudden drops in sponsor placement success.
""")

st.subheader("Download Processed Analytics Data")

download_df = filtered.copy()
download_df["date"] = download_df["date"].dt.strftime("%Y-%m-%d")

csv = download_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download processed analytics CSV",
    data=csv,
    file_name="care_transition_efficiency_analytics.csv",
    mime="text/csv"
)
