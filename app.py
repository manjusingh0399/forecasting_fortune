# forecasting_fortune_streamlit.py
# Fun, Razorpay-themed Streamlit webapp for "Forecasting Fortune"
# Features:
# - Upload or load default dataset (/mnt/data/razorpay_fy26_weekly_financial_data.csv)
# - Interactive EDA (KPIs, timeseries, distribution)
# - Prophet forecasting (choose metric & horizon)
# - Download forecast, view components, and playful UI

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from io import BytesIO

# ------------------ SETUP ------------------
st.set_page_config(page_title="Forecasting Fortune", layout="wide", page_icon="💳")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_default_data(path="/mnt/data/razorpay_fy26_weekly_financial_data.csv"):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

@st.cache_data
def generate_synthetic(n=1000, start="2023-01-01"):
    dates = pd.date_range(start=start, periods=n, freq="D")
    revenue = 200000 + np.linspace(0, 50000, n) + 20000*np.sin(np.arange(n)/30) + np.random.normal(0,8000,n)
    transactions = 5000 + np.linspace(0,1500,n) + 500*np.sin(np.arange(n)/14) + np.random.normal(0,300,n)
    profit = revenue * (0.15 + 0.05*np.sin(np.arange(n)/60)) + np.random.normal(0,3000,n)
    merchants = 15000 + np.linspace(0,4000,n) + 800*np.sin(np.arange(n)/90) + np.random.normal(0,500,n)
    df = pd.DataFrame({
        "date": dates,
        "revenue": np.round(revenue, 2),
        "transactions": np.round(transactions),
        "profit": np.round(profit, 2),
        "active_merchants": np.round(merchants)
    })
    return df

@st.cache_data
def prepare_prophet_df(df, date_col, value_col):
    d = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
    d["ds"] = pd.to_datetime(d["ds"])
    return d.sort_values("ds")

@st.cache_data
def train_prophet(df, yearly=True, weekly=False):
    m = Prophet(yearly_seasonality=yearly, weekly_seasonality=weekly)
    m.fit(df)
    return m

def to_excel_bytes(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="forecast")
    return out.getvalue()

# ------------------ HEADER ------------------
col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.title("Forecasting Fortune ✨💳 — Razorpay Revenue Playground")
    st.markdown("Turn Razorpay data into fun financial foresight. Predict, play, and plan — all with a few clicks ⚡")
with col2:
    st.image("https://em-content.zobj.net/source/animated-nsfw/341/money-with-wings_1f4b8.gif", width=80)

# ------------------ SIDEBAR ------------------
st.sidebar.header("📂 Dataset & Model Options")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = load_default_data() or generate_synthetic()
    if load_default_data() is None:
        st.sidebar.warning("Using synthetic data — no default found.")

# Column selections
with st.sidebar.expander("⚙️ Column Settings", expanded=True):
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    metric = st.selectbox("Metric to Forecast", [c for c in df.columns if c != date_col], index=0)

# Convert date column
df[date_col] = pd.to_datetime(df[date_col])
df_sorted = df.sort_values(date_col)

# ------------------ TABS ------------------
tabs = st.tabs(["Overview", "Forecast", "Model Play", "Download"])

# ------------------ OVERVIEW ------------------
with tabs[0]:
    st.header("Overview & KPIs")
    latest, prev = df_sorted.iloc[-1], df_sorted.iloc[-2]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue (latest)", f"₹{int(latest[metric]):,}", delta=f"{int(latest[metric]-prev[metric]):,}")
    k2.metric("Transactions", f"{int(latest.get('transactions',0)):,}")
    k3.metric("Profit", f"₹{int(latest.get('profit',0)):,}")
    k4.metric("Active Merchants", f"{int(latest.get('active_merchants',0)):,}")

    st.subheader("📈 Trend Visualization")
    fig = px.line(df_sorted, x=date_col, y=metric, title=f"{metric.title()} over Time", labels={date_col: "Date"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Correlation Snapshot")
    cols = [c for c in df.columns if c != date_col]
    if len(cols) >= 2:
        fig2 = px.scatter(df, x=cols[0], y=cols[1], trendline="ols")
        st.plotly_chart(fig2, use_container_width=True)

# ------------------ FORECAST ------------------
with tabs[1]:
    st.header("🚀 Forecasting Studio")
    horizon = st.slider("Forecast Horizon (weeks)", 4, 52, 26)
    yearly = st.checkbox("Yearly seasonality", True)
    weekly = st.checkbox("Weekly seasonality", False)
    if st.button("Train & Forecast"):
        with st.spinner("Training Prophet model..."):
            prophet_df = prepare_prophet_df(df, date_col, metric)
            model = train_prophet(prophet_df, yearly, weekly)
            future = model.make_future_dataframe(periods=horizon * 7)
            forecast = model.predict(future)
            st.subheader("📊 Forecast Visualization")
            fig_forecast = plot_plotly(model, forecast)
            st.plotly_chart(fig_forecast, use_container_width=True)

            st.subheader("Trend Components")
            st.pyplot(model.plot_components(forecast))
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))

            excel_bytes = to_excel_bytes(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
            st.download_button("📥 Download Forecast Excel", excel_bytes, "forecast.xlsx")

# ------------------ MODEL PLAY ------------------
with tabs[2]:
    st.header("🛠️ Model Play — Try What-Ifs!")
    colA, colB = st.columns(2)
    with colA:
        ma_window = st.number_input("Rolling Avg Window (days)", 3, 90, 7)
        df["rolling"] = df_sorted[metric].rolling(ma_window).mean()
        fig_ma = px.line(df, x=date_col, y=[metric, "rolling"], title="Raw vs Rolling Avg")
        st.plotly_chart(fig_ma, use_container_width=True)
    with colB:
        shock = st.slider("Simulate revenue shock (%)", -50, 200, 0)
        st.write(f"Future scenario adjustment: {shock}% growth (hypothetical).")

# ------------------ DOWNLOAD ------------------
with tabs[3]:
    st.header("📦 Download & Share")
    st.download_button("Download dataset (CSV)", df.to_csv(index=False).encode("utf-8"), "razorpay_dataset.csv")
    st.download_button("Download README", b"Run locally: streamlit run forecasting_fortune_streamlit.py", "README.txt")

st.markdown("---")
st.markdown("Made with ❤️ and caffeine for the *Forecasting Fortune* project ☕💳✨")
