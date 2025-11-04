# Forecasting Fortune — Razorpay Analytics (Grant Thornton Edition)
# Polished Version | Fixed interpolation issue

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly

# -----------------------------------------------
# 🎨 Page Setup & Styling
# -----------------------------------------------
st.set_page_config(
    page_title="Forecasting Fortune: Razorpay Analytics",
    layout="wide",
    page_icon="💳"
)

st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #eef2ff, #f8faff);
        }
        h1, h2, h3 {
            font-family: 'Poppins', sans-serif;
            color: #0B69FF;
        }
        .metric-card {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 3px 8px rgba(0,0,0,0.1);
            text-align: center;
            transition: 0.3s;
        }
        .metric-card:hover {
            transform: scale(1.03);
            box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            font-size: 13px;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------
# 📘 Sidebar
# -----------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Razorpay_logo.svg/2560px-Razorpay_logo.svg.png", use_container_width=True)
st.sidebar.title("💳 Forecasting Fortune")
st.sidebar.markdown("""
A **FinTech analytics simulation** for **Razorpay**, designed in collaboration with **Grant Thornton–style business insights**.

We use:
- 📊 **Financial Forecasting (Prophet)**
- 💡 **Revenue Drivers Analysis**
- 🧮 **Profitability Simulation**

To predict revenue, understand growth patterns, and drive strategic business decisions for FY2025–26.
""")

# -----------------------------------------------
# 🧾 Data Loading
# -----------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("razorpay_fy26_weekly_financial_data.csv")
    except:
        # Generate synthetic data if file missing
        dates = pd.date_range("2025-04-01", "2026-03-31", freq='W')
        np.random.seed(42)
        df = pd.DataFrame({
            "Date": dates,
            "Revenue": np.random.randint(50, 120, len(dates)) * 1e5,
            "Profit": np.random.randint(10, 40, len(dates)) * 1e4,
            "Transactions": np.random.randint(2000, 8000, len(dates)),
            "Active_Merchants": np.random.randint(100, 500, len(dates))
        })
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= "2025-04-01") & (df["Date"] <= "2026-03-31")]
    df = df.sort_values("Date").reset_index(drop=True)

    # 🧹 Ensure numeric types only for model
    numeric_cols = ["Revenue", "Profit", "Transactions", "Active_Merchants"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
    return df

df = load_data()

# -----------------------------------------------
# 🏠 Section 1: Overview Dashboard
# -----------------------------------------------
st.title("💳 Forecasting Fortune: Razorpay Revenue Analytics")
st.markdown("### A data-driven forecasting project that helps Razorpay anticipate revenue, trends, and growth for **FY2025–26**.")

st.divider()
st.subheader("🏦 Business Overview — FY25–26 Highlights")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card">💰<br><b>Total Revenue</b><br>' + f"₹{df['Revenue'].sum()/1e7:.2f} Cr</div>", unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card">📈<br><b>Avg Weekly Profit</b><br>' + f"₹{df['Profit'].mean()/1e5:.2f} L</div>", unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card">🛍️<br><b>Total Transactions</b><br>' + f"{df['Transactions'].sum():,}</div>", unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card">👩‍💼<br><b>Active Merchants</b><br>' + f"{int(df['Active_Merchants'].mean()):,}</div>", unsafe_allow_html=True)

fig = px.line(df, x="Date", y="Revenue", title="📊 Weekly Revenue Trend (FY25–26)", color_discrete_sequence=["#0B69FF"])
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------
# 🔮 Section 2: Forecasting Studio
# -----------------------------------------------
st.subheader("🔮 Forecasting Studio — Predict the Next 12 Weeks")

data = df.rename(columns={"Date": "ds", "Revenue": "y"})
model = Prophet()
model.fit(data)
future = model.make_future_dataframe(periods=12, freq="W")
forecast = model.predict(future)

fig2 = plot_plotly(model, forecast)
st.plotly_chart(fig2, use_container_width=True)

st.info("📈 Forecast Insight: The next quarter shows consistent growth with mild
