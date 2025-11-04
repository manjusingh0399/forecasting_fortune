# app_v2.py
# Forecasting Fortune — Grant Thornton Case Edition (Apr 2025 -> Mar 2026)
# Streamlit app that uses Prophet for forecasting and provides business insights.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from io import BytesIO
from datetime import datetime

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Forecasting Fortune — Grant Thornton", layout="wide", page_icon="💳")

START_DATE = pd.to_datetime("2025-04-01")
END_DATE = pd.to_datetime("2026-03-31")
FORECAST_WEEKS = 12  # forecast horizon after end of fiscal window

# -------------------- HELPERS --------------------
@st.cache_data
def load_uploaded_or_default(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.warning("Uploaded file couldn't be read as CSV. Using synthetic data instead.")
    # generate synthetic weekly data for fiscal year
    return generate_synthetic_weekly(START_DATE, END_DATE)

def generate_synthetic_weekly(start, end):
    # Create weekly date range (week start: Monday)
    dates = pd.date_range(start=start, end=end, freq='W-MON')
    n = len(dates)
    # Make realistic-ish patterns
    base_revenue = 5_000_000  # weekly rupees baseline
    trend = np.linspace(0, 800_000, n)  # gentle growth across year
    season = 400_000 * np.sin(np.arange(n) * 2 * np.pi / 26)  # semiannual-ish
    noise = np.random.normal(0, 150_000, n)
    revenue = np.round(base_revenue + trend + season + noise, 2)
    transactions = np.round(100_000 + np.linspace(0, 20_000, n) + 5_000*np.sin(np.arange(n)/4) + np.random.normal(0,2000,n))
    profit_margin = 0.12 + 0.01 * np.sin(np.arange(n)/8)  # around 12%
    profit = np.round(revenue * profit_margin + np.random.normal(0, 40_000, n), 2)
    merchants = np.round(200_000 + np.linspace(0, 10_000, n) + 3_000*np.sin(np.arange(n)/12) + np.random.normal(0,1000,n))
    df = pd.DataFrame({
        "date": dates,
        "revenue": revenue,
        "transactions": transactions,
        "profit": profit,
        "active_merchants": merchants
    })
    return df

def filter_fy_window(df, date_col="date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    mask = (df[date_col] >= START_DATE) & (df[date_col] <= END_DATE)
    df = df.loc[mask].sort_values(date_col).reset_index(drop=True)
    return df

def prepare_prophet_df(df, date_col="date", value_col="revenue"):
    d = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
    d["ds"] = pd.to_datetime(d["ds"])
    return d

def train_and_forecast(prophet_df, weeks_ahead=12, yearly=True, weekly=False):
    m = Prophet(yearly_seasonality=yearly, weekly_seasonality=weekly, daily_seasonality=False)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=weeks_ahead*7)  # days
    fc = m.predict(future)
    return m, fc

def to_excel_bytes(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="forecast")
    return out.getvalue()

# -------------------- APP UI --------------------
st.markdown("<h1 style='color:#0B69FF'>Forecasting Fortune — Razorpay (Grant Thornton Edition)</h1>", unsafe_allow_html=True)
st.markdown("**Scope:** April 1, 2025 — March 31, 2026. Forecast next 12 weeks after Mar 31, 2026.  ")

# Sidebar: upload
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional) — app will filter Apr 2025 to Mar 2026", type=["csv"])
df_raw = load_uploaded_or_default(uploaded)
df = filter_fy_window(df_raw, date_col="date")
if df.empty:
    st.error("No data found for Apr 2025 - Mar 2026. Using synthetic weekly data for that window.")
    df = generate_synthetic_weekly(START_DATE, END_DATE)

# Basic cleaning & ensure numeric
for col in ["revenue","transactions","profit","active_merchants"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
# Fill small gaps with forward fill
df = df.sort_values("date").reset_index(drop=True)
df.interpolate(method='time', inplace=True)
df.fillna(method='ffill', inplace=True)

# Tabs
tabs = st.tabs(["Overview", "Forecasting Studio", "Profitability Playbook", "Strategy Insights"])

# -------------------- OVERVIEW --------------------
with tabs[0]:
    st.subheader("Business Overview — FY 2025-26")
    # KPIs: show last available week in window
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue (latest week)", f"₹{int(latest['revenue']):,}", delta=f"₹{int(latest['revenue'] - prev['revenue']):,}")
    c2.metric("Profit (latest week)", f"₹{int(latest.get('profit',0)):,}")
    c3.metric("Transactions (latest week)", f"{int(latest.get('transactions',0)):,}")
    c4.metric("Active Merchants (latest)", f"{int(latest.get('active_merchants',0)):,}")

    st.markdown("**Revenue trend (weekly)**")
    fig_rev = px.line(df, x="date", y="revenue", title="Weekly Revenue — FY25-26", labels={"revenue":"Revenue (₹)", "date":"Week"})
    fig_rev.update_layout(hovermode="x unified")
    st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("**Other metrics**")
    fig_multi = px.line(df, x="date", y=["transactions","profit","active_merchants"], labels={"value":"Value","variable":"Metric"})
    st.plotly_chart(fig_multi, use_container_width=True)

    st.markdown("**Correlation heatmap (numeric metrics)**")
    num_cols = [c for c in ["revenue","profit","transactions","active_merchants"] if c in df.columns]
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", title="Correlation between metrics")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation.")

# -------------------- FORECASTING --------------------
with tabs[1]:
    st.subheader("Revenue Crystal Ball 🔮")
    st.markdown("We use Prophet to forecast revenue for the next 12 weeks after Mar 31, 2026.")
    # allow user to choose metric to forecast (default revenue)
    metric_choice = st.selectbox("Metric to forecast", options=[c for c in df.columns if c!="date"], index=0)
    yearly = st.checkbox("Add yearly seasonality", value=True)
    weekly = st.checkbox("Add weekly seasonality", value=False)
    if st.button("Run Forecast"):
        with st.spinner("Training model and creating forecast..."):
            prop_df = prepare_prophet_df(df, date_col="date", value_col=metric_choice)
            # Prophet wants regular spacing; it's ok with weekly/daily mixed — we use as is.
            model, forecast = train_and_forecast(prop_df, weeks_ahead=FORECAST_WEEKS, yearly=yearly, weekly=weekly)
            st.success("Forecast ready ✅")

            # Plot interactive forecast
            fig = plot_plotly(model, forecast)
            st.plotly_chart(fig.update_layout(title=f"Forecast for {metric_choice.title()}"), use_container_width=True)

            # Components
            st.markdown("**Forecast components**")
            comp_fig = model.plot_components(forecast)
            st.pyplot(comp_fig)

            # show forecast tail (period after the end date)
            future_mask = forecast["ds"] > END_DATE
            ftail = forecast.loc[future_mask, ["ds","yhat","yhat_lower","yhat_upper"]].copy()
            ftail["ds"] = pd.to_datetime(ftail["ds"]).dt.date
            st.dataframe(ftail.head(FORECAST_WEEKS*7).head(FORECAST_WEEKS))  # show weekly head rows approx

            # Simple summary: predicted % change from last actual to median of forecast first 4 weeks
            if not ftail.empty:
                first_n = ftail.head(7)  # approx 1 week (model predicts daily)
                # compute mean yhat for the first 7 days => weekly estimate
                pred_weekly = first_n["yhat"].mean()
                last_actual = df[metric_choice].iloc[-1]
                pct_change = (pred_weekly - last_actual) / last_actual * 100 if last_actual != 0 else np.nan
                st.metric("Predicted weekly change vs last actual", f"{pct_change:.2f}%")

            # download
            st.download_button("Download full forecast (Excel)", to_excel_bytes(forecast), file_name="forecast_full.xlsx")

# -------------------- PROFITABILITY PLAYBOOK --------------------
with tabs[2]:
    st.subheader("Profitability Playbook — What-if Scenarios")
    st.markdown("Simulate improvements in profit margin or transaction growth and see impact on revenue/profit.")
    colA, colB = st.columns(2)
    with colA:
        profit_margin_delta = st.slider("Change in profit margin (percentage points)", -10.0, 30.0, 0.0, step=0.5)
        st.write("This adjusts the profit margin applied to revenue (hypothetical).")
        # compute baseline margin from data
        if ("profit" in df.columns) and ("revenue" in df.columns):
            baseline_margin = (df["profit"].sum() / df["revenue"].sum()) if df["revenue"].sum()!=0 else 0.12
        else:
            baseline_margin = 0.12
        new_margin = baseline_margin + profit_margin_delta/100.0
        st.write(f"Baseline margin ≈ {baseline_margin*100:.2f}%, New margin ≈ {new_margin*100:.2f}%")
        df_play = df.copy()
        df_play["sim_profit"] = df_play["revenue"] * new_margin
        fig_sim = px.line(df_play, x="date", y=["profit","sim_profit"], labels={"value":"Profit (₹)","variable":"Series"}, title="Actual Profit vs Simulated Profit")
        st.plotly_chart(fig_sim, use_container_width=True)
        total_gain = df_play["sim_profit"].sum() - df_play["profit"].sum() if "profit" in df.columns else df_play["sim_profit"].sum()
        st.metric("Estimated incremental profit (FY)", f"₹{int(total_gain):,}")

    with colB:
        tx_growth = st.slider("Simulate Transactions growth (%)", -50, 200, 0)
        st.write("This simulates percentage growth in transactions across the FY window.")
        df_tx = df.copy()
        if "transactions" in df.columns:
            df_tx["sim_transactions"] = df_tx["transactions"] * (1 + tx_growth/100.0)
            # naive conversion: revenue per transaction baseline
            rev_per_tx = (df["revenue"].sum() / df["transactions"].sum()) if df["transactions"].sum()!=0 else 50
            df_tx["sim_revenue"] = df_tx["sim_transactions"] * rev_per_tx
            fig_tx = px.line(df_tx, x="date", y=["revenue","sim_revenue"], labels={"value":"Revenue (₹)","variable":"Series"}, title="Revenue vs Transaction-driven Revenue")
            st.plotly_chart(fig_tx, use_container_width=True)
            incremental = df_tx["sim_revenue"].sum() - df["revenue"].sum()
            st.metric("Estimated incremental revenue (FY)", f"₹{int(incremental):,}")
        else:
            st.info("No 'transactions' column found in dataset to simulate transaction growth.")

# -------------------- STRATEGY INSIGHTS --------------------
with tabs[3]:
    st.subheader("Strategy Insights — Executive Summary")
    st.markdown("Automatic data-driven takeaways (use these in your Grant Thornton-style brief).")

    # Basic driver analysis: correlation coefficients
    driver_text = ""
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        # find top absolute correlation with revenue (excluding revenue self)
        if "revenue" in corr.columns:
            corr_with_rev = corr["revenue"].drop("revenue", errors="ignore").abs().sort_values(ascending=False)
            if not corr_with_rev.empty:
                top_driver = corr_with_rev.index[0]
                top_val = corr_with_rev.iloc[0]
                driver_text += f"- Top driver correlated with revenue: **{top_driver}** (|r| = {top_val:.2f}).\n"
            else:
                driver_text += "- Correlation analysis inconclusive.\n"
    else:
        driver_text += "- Not enough metrics to run driver analysis.\n"

    # Growth rate across FY
    try:
        start_val = df["revenue"].iloc[0]
        end_val = df["revenue"].iloc[-1]
        yoy_growth = (end_val - start_val) / start_val * 100 if start_val != 0 else np.nan
        driver_text += f"- Revenue growth across FY window: **{yoy_growth:.2f}%** (first → last week).\n"
    except Exception:
        driver_text += "- Could not compute growth (missing revenue values).\n"

    # Forecast hint (if forecast exists in session state)
    # We'll check if user ran forecast by looking for last forecast in downloads? Simpler: re-run small forecast quickly for a 4-week horizon to produce a direction hint.
    try:
        pf = prepare_prophet_df(df, "date", "revenue")
        mtmp = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        mtmp.fit(pf)
        fut = mtmp.make_future_dataframe(periods=28)  # 4 weeks
        ftmp = mtmp.predict(fut)
        # compute mean yhat in first 7 days after END_DATE
        post = ftmp[ftmp["ds"] > END_DATE]
        if not post.empty:
            next_week_mean = post.head(7)["yhat"].mean()
            last_actual = df["revenue"].iloc[-1]
            pct_change = (next_week_mean - last_actual)/last_actual*100 if last_actual!=0 else 0.0
            driver_text += f"- Short-term forecast indicates a weekly revenue change of **{pct_change:.2f}%** vs last week.\n"
    except Exception:
        driver_text += "- Forecast quick-check unavailable (model error).\n"

    # Optimization suggestions (simple rule-based)
    suggestions = []
    if "active_merchants" in df.columns and "transactions" in df.columns:
        # If active merchants correlate strongly with revenue, suggest merchant acquisition
        corr_am_rev = df["active_merchants"].corr(df["revenue"])
        if abs(corr_am_rev) > 0.4:
            suggestions.append("Prioritize merchant acquisition & retention programs — merchants show strong correlation to revenue.")
    if "transactions" in df.columns:
        tx_var = df["transactions"].pct_change().std()
        if tx_var > 0.1:
            suggestions.append("Stabilize transaction volumes (promotions/merchant support) to reduce revenue volatility.")
    # add margin suggestion
    if "profit" in df.columns and "revenue" in df.columns:
        margin = df["profit"].sum() / df["revenue"].sum() if df["revenue"].sum()!=0 else 0.12
        if margin < 0.12:
            suggestions.append("Focus on cost optimization (payment routing, processing fees) to improve profit margins.")

    # Render text
    st.markdown(driver_text)
    if suggestions:
        st.markdown("**Suggested focus areas:**")
        for s in suggestions:
            st.markdown(f"- {s}")
    else:
        st.markdown("- No specific optimization suggestions from heuristics. Consider deeper segmentation analysis.")

    st.markdown("---")
    st.markdown("**Use these insights for your slide deck:** copy the bullets above into a 1-slide executive summary for leadership.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Forecasting Fortune — Built for the Grant Thornton Razorpay case. Scope limited to Apr 2025 → Mar 2026. Made with ❤️ and data.")

