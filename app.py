from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "ds_salaries_clean.csv"

st.set_page_config(page_title="Tech Salary Intelligence", page_icon="$", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1500px; padding-top: 1.5rem; }
    [data-testid="stMetric"] { background: #f5f7f6; border: 1px solid #dce5e1; padding: 1rem; border-radius: 8px; }
    [data-testid="stMetricValue"] { color: #124e4a; }
    h1, h2, h3 { color: #124e4a; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model(data):
    features = [
        "work_year", "experience_level", "employment_type", "job_title",
        "remote_ratio", "company_size", "employee_continent", "company_continent",
    ]
    categorical = [
        "experience_level", "employment_type", "job_title", "company_size",
        "employee_continent", "company_continent",
    ]
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", ["work_year", "remote_ratio"]),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
        )),
    ])
    model.fit(data[features], data["salary_in_usd"])
    return model


@st.cache_data
def historical_summary(data):
    yearly = data.groupby("work_year")["salary_in_usd"].median()
    first, last = yearly.iloc[0], yearly.iloc[-1]
    periods = yearly.index[-1] - yearly.index[0]
    growth = 0 if periods == 0 else (last / first) ** (1 / periods) - 1
    return yearly, growth


def level_label(value):
    return {"EN": "Entry-level", "MI": "Mid-level", "SE": "Senior-level", "EX": "Executive"}.get(value, value)


def employment_label(value):
    return {"FT": "Full-time", "PT": "Part-time", "CT": "Contract", "FL": "Freelance"}.get(value, value)


def money(value):
    return f"${value:,.0f}"


df = load_data()
model = train_model(df)
yearly_medians, growth_rate = historical_summary(df)
latest_year = int(df["work_year"].max())

st.title("Tech Salary Intelligence")
st.caption("A practical salary benchmark and future-scenario tool for global technology roles.")
st.warning("The source data covers 2020-2022 and is globally imbalanced. Forecasts are scenarios, not guarantees.")

with st.sidebar:
    st.header("Build a job scenario")
    job_title = st.selectbox("Role", sorted(df["job_title"].unique()))
    experience = st.selectbox("Experience", ["EN", "MI", "SE", "EX"], format_func=level_label)
    employment = st.selectbox("Employment", sorted(df["employment_type"].unique()), format_func=employment_label)
    company_size = st.selectbox(
        "Company size", sorted(df["company_size"].unique()),
        format_func=lambda value: {"S": "Small", "M": "Medium", "L": "Large"}.get(value, value),
    )
    continents = sorted(df["company_continent"].unique())
    continent_default = continents.index("North America") if "North America" in continents else 0
    company_continent = st.selectbox("Company continent", continents, index=continent_default)
    employee_continent = st.selectbox("Employee continent", sorted(df["employee_continent"].unique()), index=continent_default)
    remote_ratio = st.selectbox(
        "Work arrangement", [0, 50, 100],
        format_func=lambda value: {0: "On-site", 50: "Hybrid", 100: "Fully remote"}[value],
    )
    future_year = st.slider("Target forecast year", latest_year, latest_year + 10, latest_year + 5)

profile = pd.DataFrame([{
    "work_year": latest_year,
    "experience_level": experience,
    "employment_type": employment,
    "job_title": job_title,
    "remote_ratio": remote_ratio,
    "company_size": company_size,
    "employee_continent": employee_continent,
    "company_continent": company_continent,
}])

base_salary = float(model.predict(profile)[0])
transformed_profile = model.named_steps["preprocessor"].transform(profile)
trees = model.named_steps["regressor"].estimators_
tree_predictions = np.array([tree.predict(transformed_profile)[0] for tree in trees])
base_low, base_high = np.percentile(tree_predictions, [10, 90])
years_ahead = future_year - latest_year
forecast_salary = base_salary * (1 + growth_rate) ** years_ahead
forecast_low = base_low * (1 + growth_rate) ** years_ahead
forecast_high = base_high * (1 + growth_rate) ** years_ahead

role_data = df[df["job_title"] == job_title]
role_median = role_data["salary_in_usd"].median()
role_low, role_high = role_data["salary_in_usd"].quantile([0.25, 0.75])
role_percentile = (role_data["salary_in_usd"] <= base_salary).mean()

st.subheader("Your scenario")
scenario_text = f"{level_label(experience)} {job_title} | {employment_label(employment)} | {company_continent} company | {remote_ratio}% remote"
st.caption(scenario_text)

kpis = st.columns(5)
kpis[0].metric("Estimated 2022 salary", money(base_salary))
kpis[1].metric(f"Projected {future_year}", money(forecast_salary), f"{forecast_salary / base_salary - 1:.1%}")
kpis[2].metric("Model range", f"{money(base_low)} - {money(base_high)}")
kpis[3].metric("Role records", f"{len(role_data):,}")
kpis[4].metric("Position in role data", f"{role_percentile:.0%}")

st.info(
    f"For {job_title} records, the observed middle 50% is {money(role_low)} to {money(role_high)}. "
    f"The model estimate is at the {role_percentile:.0%} percentile of this role's observed salaries."
)

st.subheader("Forecast with uncertainty")
forecast_years = np.arange(latest_year, future_year + 1)
forecast_frame = pd.DataFrame({
    "Year": forecast_years,
    "Estimate": [base_salary * (1 + growth_rate) ** (year - latest_year) for year in forecast_years],
    "Lower range": [base_low * (1 + growth_rate) ** (year - latest_year) for year in forecast_years],
    "Upper range": [base_high * (1 + growth_rate) ** (year - latest_year) for year in forecast_years],
})
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast_frame["Year"], y=forecast_frame["Upper range"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=forecast_frame["Year"], y=forecast_frame["Lower range"], fill="tonexty", fillcolor="rgba(18,78,74,0.16)", line=dict(width=0), name="Model range"))
fig.add_trace(go.Scatter(x=forecast_frame["Year"], y=forecast_frame["Estimate"], mode="lines+markers", line=dict(color="#124e4a", width=4), name="Scenario estimate"))
historical_frame = yearly_medians.reset_index()
fig.add_trace(go.Scatter(x=historical_frame["work_year"], y=historical_frame["salary_in_usd"], mode="lines+markers", line=dict(color="#e07a5f", width=3), name="Observed market median"))
fig.update_layout(title="Observed market and selected scenario", xaxis_title="Year", yaxis_title="Annual salary (USD)", hovermode="x unified", height=440)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"The shaded band is the 10th-90th percentile of individual forest-tree estimates, compounded using {growth_rate:.1%} historical annual median growth.")

market_tab, role_tab, geography_tab = st.tabs(["Market structure", "Role benchmark", "Geography and work mode"])

with market_tab:
    st.subheader("Demand versus pay")
    demand_pay = (
        df.groupby("job_title", as_index=False)["salary_in_usd"]
        .agg(median_salary="median", job_count="count")
        .query("job_count >= 3")
    )
    demand_chart = px.scatter(
        demand_pay, x="job_count", y="median_salary", size="job_count", color="median_salary",
        hover_name="job_title", color_continuous_scale="Tealgrn",
        labels={"job_count": "Number of records", "median_salary": "Median salary (USD)"},
        title="Which roles combine demand and pay?",
    )
    demand_chart.add_vline(x=len(role_data), line_dash="dot", line_color="#e07a5f", annotation_text=f"Selected role: {len(role_data)} records")
    demand_chart.update_layout(height=500, coloraxis_colorbar_title="Median USD")
    st.plotly_chart(demand_chart, use_container_width=True)
    st.caption("Roles farther right are more common in this dataset. Roles higher on the chart have higher median salaries. This is descriptive, not causal.")

with role_tab:
    st.subheader(f"{job_title}: observed salary benchmark")
    role_hist = px.histogram(
        role_data, x="salary_in_usd", nbins=20, marginal="box",
        labels={"salary_in_usd": "Annual salary (USD)"}, title=f"Salary distribution for {job_title}",
    )
    role_hist.add_vline(x=base_salary, line_color="#124e4a", line_width=3, annotation_text="Model estimate")
    role_hist.add_vline(x=role_median, line_color="#e07a5f", line_dash="dash", annotation_text="Role median")
    role_hist.update_layout(height=430)
    st.plotly_chart(role_hist, use_container_width=True)
    role_summary = pd.DataFrame({
        "Measure": ["Records", "25th percentile", "Median", "75th percentile", "Model estimate"],
        "Salary": [len(role_data), role_low, role_median, role_high, base_salary],
    })
    st.dataframe(role_summary, use_container_width=True, hide_index=True)

with geography_tab:
    st.subheader("Geography and work arrangement")
    geo_left, geo_right = st.columns(2)
    with geo_left:
        geo = df.groupby("company_continent", as_index=False)["salary_in_usd"].agg(median_salary="median", jobs="count").sort_values("median_salary")
        geo_chart = px.bar(geo, x="median_salary", y="company_continent", orientation="h", text="jobs", color="median_salary", color_continuous_scale="Tealgrn", labels={"median_salary": "Median salary (USD)", "company_continent": ""}, title="Salary by company continent")
        geo_chart.update_traces(texttemplate="n=%{text}", textposition="outside")
        geo_chart.update_layout(height=420, coloraxis_showscale=False)
        st.plotly_chart(geo_chart, use_container_width=True)
    with geo_right:
        remote = df.groupby("remote_name", as_index=False)["salary_in_usd"].agg(median_salary="median", jobs="count")
        remote_chart = px.bar(remote, x="remote_name", y="median_salary", text="jobs", color="median_salary", color_continuous_scale="Sunsetdark", labels={"median_salary": "Median salary (USD)", "remote_name": ""}, title="Salary by work arrangement")
        remote_chart.update_traces(texttemplate="n=%{text}", textposition="outside")
        remote_chart.update_layout(height=420, coloraxis_showscale=False)
        st.plotly_chart(remote_chart, use_container_width=True)

with st.expander("Methodology and limitations"):
    st.markdown(
        """
        **Model:** Random forest regression with one-hot encoding for categorical variables.

        **Forecast:** The model estimates a comparable salary for 2022. That estimate and its tree-based range are then projected using the historical annual growth in the dataset's median salary.

        **Important limitations:** The data is global, ends in 2022, contains uneven geographic samples, and has no skills, industry, company, or benefits information. Future projections should be treated as scenarios for comparison, not promises.
        """
    )
