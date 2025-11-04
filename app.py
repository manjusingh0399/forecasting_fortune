# app.py — Forecasting Fortune 💳
# Razorpay-themed revenue forecasting & analytics webapp
# Safe, self-contained, Streamlit Cloud ready

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from io import BytesIO
import base64

# --------------------------- PAGE CONFIG ---------------------------
st.set_page_config(
    page_title="Forecasting Fortune 💳",
    layout="wide",
    page_icon="💫",
    initial_sidebar_state="expanded"
)

# --------------------------- THEME ---------------------------
st.markdown("""
    <style>
    body { background-color: #f8faff; }
    .main {
        background: linear-gradient(180deg, rgba(240,245,255,1) 0%, rgba(255,255,255,1) 100%);
        border-radius: 20px;
        padding: 20px;
    }
    .stApp header { display: none; }
    .block-container { padding-top: 1rem; }
    h1, h2, h3, h4 { color: #0052cc; }
    </style>
""", unsafe_allow_html=True)

# --------------------------- HELPERS ---------------------------
@st.cache_data
def generate_data(n=1000, start="2023-01-01"):
    dates = pd.date_range(start=start, periods=n, freq="D")
    revenue = 200000 + np.linspace(0, 50000, n) + 20000*np.sin(np.arange(n)/30) + np.random.normal(0, 8000, n)
    profit = revenue * (0.18 + 0.02*np.sin(np.arange(n)/50)) + np.random.normal(0, 2000, n)
    transactions = 5000 + np.linspace(0, 1000, n) + 400*np.sin(np.arange(n)/14) + np.random.normal(0, 200, n)
    merchants = 12000 + np.linspace(0, 3000, n) + 700*np.sin(np.arange(n)/90) + np.random.normal(0, 400, n)
    df = pd.DataFrame({
        "date": dates,
        "revenue": np.round(revenue, 2),
        "profit": np.round(profit, 2),
        "transactions": np.round(transactions),
        "active_merchants": np.round(merchants)
    })
    return df

@st.cache_data
def prepare_prophet(df, date_col, metric):
    d = df[[date_col, metric]].rename(columns={date_col: "ds", metric: "y"})
    d["ds"] = pd.to_datetime(d["ds"])
    return d

@st.cache_data
def train_prophet_model(df, yearly=True, weekly=False):
    model = Prophet(yearly_seasonality=yearly, weekly_seasonality=weekly)
    model.fit(df)
    return model

def download_excel(df):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Forecast")
    return bio.getvalue()

# --------------------------- SIDEBAR ---------------------------
st.sidebar.image("https://razorpay.com/blog/content/images/2022/07/logo-1.png", width=160)
st.sidebar.header("⚙️ Setup")

uploaded = st.sidebar.file_uploader("Upload your Razorpay CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = generate_data()

date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
metric = st.sidebar.selectbox("Select Metric to Forecast", [c for c in df.columns if c != date_col], index=0)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Try uploading your own transaction data for personalized forecasting!")

# --------------------------- MAIN TITLE ---------------------------
st.title("💳 Forecasting Fortune — Razorpay Revenue Analytics")
st.markdown("""
Transforming raw financial data into predictive business intelligence.  
Explore trends, forecast revenue, and visualize growth — the Razorpay way. ⚡
""")

tabs = st.tabs(["📊 Overview", "📈 Forecasting", "🧠 Model Play", "📥 Download"])

# --------------------------- OVERVIEW TAB ---------------------------
with tabs[0]:
    st.header("📊 Business Overview")

    k1, k2, k3, k4 = st.columns(4)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    k1.metric("Revenue", f"₹{int(latest['revenue']):,}", delta=f"{int(latest['revenue'] - prev['revenue']):,}")
    k2.metric("Profit", f"₹{int(latest['profit']):,}")
    k3.metric("Transactions", f"{int(latest['transactions']):,}")
    k4.metric("Active Merchants", f"{int(latest['active_merchants']):,}")

    fig = px.line(df, x="date", y=metric, title=f"{metric.title()} Trend Over Time", color_discrete_sequence=["#0052cc"])
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Insights")
    num_cols = [c for c in df.columns if c != date_col]
    if len(num_cols) > 1:
        fig_corr = px.imshow(df[num_cols].corr(), text_auto=True, color_continuous_scale="Blues", title="Metric Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)

# --------------------------- FORECAST TAB ---------------------------
with tabs[1]:
    st.header("🔮 Prophet Forecasting Studio")

    horizon = st.slider("Forecast Horizon (weeks)", 4, 52, 26)
    yearly = st.checkbox("Include yearly seasonality", True)
    weekly = st.checkbox("Include weekly seasonality", False)

    if st.button("🚀 Run Forecast"):
        with st.spinner("Training Prophet model..."):
            df_prophet = prepare_prophet(df, date_col, metric)
            model = train_prophet_model(df_prophet, yearly, weekly)
            future = model.make_future_dataframe(periods=horizon * 7)
            forecast = model.predict(future)

            st.success("Forecast complete ✅")
            fig_forecast = plot_plotly(model, forecast)
            st.plotly_chart(fig_forecast, use_container_width=True)

            st.subheader("Forecast Components")
            st.pyplot(model.plot_components(forecast))

            st.subheader("Predicted Values (last few)")
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))

            excel = download_excel(forecast)
            st.download_button("📥 Download Forecast (Excel)", excel, "forecast.xlsx")

# --------------------------- MODEL PLAY TAB ---------------------------
with tabs[2]:
    st.header("🧠 Model Play — Explore Scenarios")

    colA, colB = st.columns(2)
    with colA:
        window = st.number_input("Rolling Average (days)", 3, 60, 7)
        df["rolling"] = df[metric].rolling(window).mean()
        fig_roll = px.line(df, x="date", y=["rolling", metric],
                           labels={"value": "Value", "variable": "Series"},
                           title="Rolling Average vs Actual")
        st.plotly_chart(fig_roll, use_container_width=True)
    with colB:
        shock = st.slider("Simulate % Growth Shock", -50, 200, 0)
        adj_df = df.copy()
        adj_df["adjusted"] = adj_df[metric] * (1 + shock / 100)
        fig_adj = px.line(adj_df, x="date", y=["adjusted", metric],
                          title=f"{shock}% Shock Scenario on {metric.title()}")
        st.plotly_chart(fig_adj, use_container_width=True)

# --------------------------- DOWNLOAD TAB ---------------------------
with tabs[3]:
    st.header("📥 Downloads & Resources")
    st.download_button("Download Dataset (CSV)", df.to_csv(index=False).encode(), "razorpay_dataset.csv")
    st.download_button("Download App README", "Run locally: streamlit run app.py".encode(), "README.txt")
    st.markdown("Built for learning, insight, and a little fintech fun ✨")

st.markdown("---")
st.caption("💳 Forecasting Fortune | A Razorpay-themed analytics experience made with ❤️ + data.")
