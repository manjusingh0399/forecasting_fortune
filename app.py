import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Forecasting Fortune 💸 | Razorpay Profitability Dashboard",
    page_icon="💳",
    layout="wide"
)

# -----------------------------------------------------------
# HEADER
# -----------------------------------------------------------
st.markdown("""
    <style>
        .main-title {
            font-size:42px;
            font-weight:800;
            color:#0C6CF2;
            text-align:center;
            padding:15px 0;
        }
        .sub-text {
            text-align:center;
            color:#555;
            font-size:18px;
            margin-bottom:25px;
        }
        .card {
            background-color:#f7f9ff;
            border-radius:15px;
            padding:18px;
            text-align:center;
            box-shadow: 0px 3px 8px rgba(0,0,0,0.08);
        }
        .insight-box {
            background-color:#f0f7ff;
            border-radius:10px;
            padding:15px;
            margin-top:10px;
        }
    </style>
    <div class="main-title">💸 Forecasting Fortune: Razorpay FY25–26</div>
    <div class="sub-text">Analytics • Forecasting • Profitability Optimization ⚡</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
try:
    df = pd.read_csv("razorpay_fy26_weekly_financial_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
except:
    st.warning("⚠️ Could not find your CSV file. Generating synthetic data for demo.")
    dates = pd.date_range(start='2025-04-01', end='2026-03-31', freq='W')
    df = pd.DataFrame({
        'Date': dates,
        'Transaction_Value': np.random.randint(5_00_00_000, 10_00_00_000, len(dates)),
        'Revenue': np.random.randint(5_00_000, 20_00_000, len(dates)),
        'Growth_Rate': np.random.uniform(1, 5, len(dates))
    })

# filter for FY25–26
df = df[(df['Date'] >= '2025-04-01') & (df['Date'] <= '2026-03-31')]
df = df.sort_values('Date')

# -----------------------------------------------------------
# PROFIT CALCULATION
# -----------------------------------------------------------
np.random.seed(42)
df['Operating_Cost'] = df['Revenue'] * np.random.uniform(0.6, 0.8, len(df))
df['Profit'] = df['Revenue'] - df['Operating_Cost']
df['Profit_Margin'] = (df['Profit'] / df['Revenue']) * 100

# -----------------------------------------------------------
# KPIs
# -----------------------------------------------------------
total_txn = df['Transaction_Value'].sum()
avg_growth = df['Growth_Rate'].mean()
avg_profit_margin = df['Profit_Margin'].mean()
total_revenue = df['Revenue'].sum()
total_profit = df['Profit'].sum()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='card'><h3>Total Transaction Value</h3><h2>₹{total_txn:,.0f}</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='card'><h3>Total Revenue</h3><h2>₹{total_revenue:,.0f}</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='card'><h3>Total Profit</h3><h2>₹{total_profit:,.0f}</h2></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='card'><h3>Avg Profit Margin</h3><h2>{avg_profit_margin:.2f}%</h2></div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------------
# VISUALS
# -----------------------------------------------------------
st.subheader("📈 Weekly Revenue & Profit Trend")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df['Date'], y=df['Revenue'], name='Revenue', line=dict(color='#0C6CF2', width=3)))
fig1.add_trace(go.Scatter(x=df['Date'], y=df['Profit'], name='Profit', line=dict(color='#00B894', width=3)))
fig1.update_layout(title="Revenue vs Profit Over Time", template='plotly_white')
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🔥 Profit Margin by Month")
df['Month'] = df['Date'].dt.strftime('%b')
monthly_margin = df.groupby('Month')['Profit_Margin'].mean().reset_index()
fig2 = px.bar(monthly_margin, x='Month', y='Profit_Margin', color='Profit_Margin',
              color_continuous_scale='Blues', title="Monthly Profit Margin (%)")
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------------
# PROFITABILITY ACCELERATOR DASHBOARD
# -----------------------------------------------------------
st.markdown("## 💼 Profitability Accelerator — 'What If' Simulation")

st.write("Adjust the sliders below to simulate cost and revenue changes for FY25–26.")

colA, colB = st.columns(2)
with colA:
    cost_reduction = st.slider("Reduce Operating Costs by (%)", 0, 30, 10)
with colB:
    revenue_growth = st.slider("Increase Revenue by (%)", 0, 25, 5)

# simulate new values
df['Optimized_Revenue'] = df['Revenue'] * (1 + revenue_growth / 100)
df['Optimized_Cost'] = df['Operating_Cost'] * (1 - cost_reduction / 100)
df['Optimized_Profit'] = df['Optimized_Revenue'] - df['Optimized_Cost']
df['Optimized_Margin'] = (df['Optimized_Profit'] / df['Optimized_Revenue']) * 100

# visuals
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df['Date'], y=df['Profit_Margin'], mode='lines', name='Current Margin', line=dict(color='#FF5252', width=3)))
fig3.add_trace(go.Scatter(x=df['Date'], y=df['Optimized_Margin'], mode='lines', name='Optimized Margin', line=dict(color='#00B894', width=3, dash='dot')))
fig3.update_layout(title="Profit Margin: Before vs After Optimization", template='plotly_white')
st.plotly_chart(fig3, use_container_width=True)

# insights
current_margin = df['Profit_Margin'].mean()
new_margin = df['Optimized_Margin'].mean()
gain = new_margin - current_margin
new_profit = df['Optimized_Profit'].sum()
profit_increase = ((new_profit - total_profit) / total_profit) * 100

st.markdown(f"""
<div class='insight-box'>
<h4>📊 Profitability Insights:</h4>
<ul>
<li>Reducing costs by <b>{cost_reduction}%</b> and increasing revenue by <b>{revenue_growth}%</b> improves overall profit margin from <b>{current_margin:.2f}%</b> to <b>{new_margin:.2f}%</b>.</li>
<li>Total annual profit rises by <b>{profit_increase:.2f}%</b> — demonstrating how small operational improvements create major financial impact.</li>
<li>Maintaining this structure could accelerate profitability by nearly <b>{gain:.2f} margin points</b> within one fiscal year.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# FINAL FEEDBACK SUMMARY
# -----------------------------------------------------------
st.markdown("## 🧠 Strategic Recommendations")
st.success(f"""
- **Cost Optimization:** Focus on reducing operating expenses from 75% → 65%. Automate payment ops, cloud optimization, and vendor rationalization.  
- **Revenue Efficiency:** Improve monetization through new merchant segments and fintech cross-products (credit, payroll, payouts).  
- **Operational Consistency:** Monitor weekly variance and trigger alerts if margins drop below 15%.  
- **Scalability:** The simulation shows profit can rise by ~{profit_increase:.2f}% with modest efficiency efforts — proving that scale + discipline = sustainability.  
- **Next Steps:** Integrate AI forecasting (Prophet/ARIMA) for quarterly trend prediction and dynamic goal-setting dashboards.
""")

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
💳 Razorpay Revenue Intelligence Dashboard | Built with ❤️ + Streamlit + Plotly
</p>
""", unsafe_allow_html=True)
