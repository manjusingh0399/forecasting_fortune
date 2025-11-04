# ==========================================================
# Forecasting Fortune — Razorpay-style Streamlit Web App
# FY25–26 (April 2025 – March 2026)
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from io import BytesIO

# ----------------------------------------------------------
# Page setup
# ----------------------------------------------------------
st.set_page_config(page_title="Forecasting Fortune 💳", page_icon="💫", layout="wide")

st.title("💫 Forecasting Fortune — Data but Make It Fashion")
st.caption("A Razorpay-inspired analytics dashboard for FY 2025–26 📊")

st.markdown("""
This fun financial-forecast app re-imagines analytics through the Razorpay lens.  
You can **analyze weekly revenue, profit, transactions & active merchants** between  
**April 2025 → March 2026** — and even simulate next-week predictions ✨  
""")

# ----------------------------------------------------------
# Helper: synthetic dataset generator
# ----------------------------------------------------------
def generate_synthetic_data():
    dates = pd.date_range("2025-04-01", "2026-03-31", freq="W")
    n = len(dates)
    np.random.seed(42)

    revenue = 5_000_000 + np.linspace(0, 800_000, n) + np.random.normal(0, 120_000, n)
    profit = 0.15 * revenue + np.random.normal(0, 40_000, n)
    transactions = 100_000 + np.linspace(0, 20000, n) + np.random.normal(0, 3000, n)
    merchants = 200_000 + np.linspace(0, 15000, n) + np.random.normal(0, 2000, n)

    df = pd.DataFrame({
        "Date": dates,
        "Revenue": np.round(revenue, 2),
        "Profit": np.round(profit, 2),
        "Transactions": np.round(transactions, 0),
        "Active Merchants": np.round(merchants, 0)
    })
    return df

# ----------------------------------------------------------
# Load or create data
# ----------------------------------------------------------
uploaded = st.sidebar.file_uploader("📂 Upload your CSV (optional)", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("✅ File uploaded successfully!")
    except Exception:
        st.sidebar.error("⚠️ Could not read CSV. Using sample data instead.")
        df = generate_synthetic_data()
else:
    df = generate_synthetic_data()

# ----------------------------------------------------------
# Data cleaning & filtering
# ----------------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df[(df["Date"] >= "2025-04-01") & (df["Date"] <= "2026-03-31")]

# ----------------------------------------------------------
# KPIs
# ----------------------------------------------------------
st.subheader("✨ Quick Financial Snapshot")
latest = df.iloc[-1]
prev = df.iloc[-2]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue", f"₹{latest['Revenue']:,.0f}", f"₹{latest['Revenue']-prev['Revenue']:,.0f}")
c2.metric("Profit", f"₹{latest['Profit']:,.0f}")
c3.metric("Transactions", f"{latest['Transactions']:,.0f}")
c4.metric("Active Merchants", f"{latest['Active Merchants']:,.0f}")

# ----------------------------------------------------------
# Visualizations
# ----------------------------------------------------------
st.markdown("### 📈 Weekly Trends")

tab1, tab2, tab3 = st.tabs(["Revenue & Profit", "Transactions", "Active Merchants"])

with tab1:
    fig1 = px.line(df, x="Date", y=["Revenue", "Profit"], title="Revenue vs Profit (FY25–26)")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = px.area(df, x="Date", y="Transactions", title="Weekly Transactions Volume", color_discrete_sequence=["#636EFA"])
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = px.bar(df, x="Date", y="Active Merchants", title="Active Merchants Trend", color="Active Merchants")
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------
# Forecast Simulation (simple linear trend)
# ----------------------------------------------------------
st.markdown("### 🔮 Forecast Simulation (Next 12 Weeks)")

# Simple trend projection
trend_weeks = 12
x = np.arange(len(df))
future_x = np.arange(len(df), len(df) + trend_weeks)
future_dates = pd.date_range(df["Date"].iloc[-1] + pd.Timedelta(weeks=1), periods=trend_weeks, freq="W")

def forecast_series(series):
    coef = np.polyfit(x, series, 1)
    return np.polyval(coef, future_x)

future_df = pd.DataFrame({
    "Date": future_dates,
    "Revenue": forecast_series(df["Revenue"]),
    "Profit": forecast_series(df["Profit"]),
    "Transactions": forecast_series(df["Transactions"]),
    "Active Merchants": forecast_series(df["Active Merchants"])
})

figf = px.line(future_df, x="Date", y="Revenue", title="Projected Revenue (Next 12 Weeks)", color_discrete_sequence=["#FF6692"])
st.plotly_chart(figf, use_container_width=True)

# ----------------------------------------------------------
# Download section
# ----------------------------------------------------------
st.markdown("### 📥 Download Data")
buffer = BytesIO()
df.to_csv(buffer, index=False)
st.download_button("Download FY25–26 Data (CSV)", data=buffer.getvalue(),
                   file_name="razorpay_forecasting_fortune_FY25_26.csv",
                   mime="text/csv")

# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------
st.markdown("---")
st.markdown("""
💡 **About this app:**  
“Forecasting Fortune” re-imagines analytics in Razorpay style — playful, modern, and data-driven.  
It demonstrates **how forecasting, visualization, and financial storytelling** can come together  
to make business analytics engaging, insightful, and fashionably fun 💅  
""")
