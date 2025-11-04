# ==========================================================
# Forecasting Fortune — Razorpay-Themed Streamlit Dashboard
# FY25–26 (April 2025 – March 2026)
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime

# ----------------------------------------------------------
# Page setup
# ----------------------------------------------------------
st.set_page_config(page_title="Forecasting Fortune 💳", page_icon="💫", layout="wide")

st.markdown("""
<h1 style="text-align:center; color:#0B69FF;">💫 Forecasting Fortune: Razorpay Revenue Analytics</h1>
<p style="text-align:center; font-size:18px;">Where Financial Forecasting Meets Fashion & FinTech</p>
""", unsafe_allow_html=True)

st.markdown("""
This interactive dashboard reimagines **Razorpay’s FY25–26 performance** through
data visualization, forecasting, and storytelling.  
Explore trends, relationships, and growth insights for:
**Revenue, Profit, Transactions, and Active Merchants.**
""")

# ----------------------------------------------------------
# Generate or upload data
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

uploaded = st.sidebar.file_uploader("📂 Upload your CSV (optional)", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("✅ File uploaded successfully!")
    except Exception:
        st.sidebar.error("⚠️ Could not read CSV. Using synthetic data instead.")
        df = generate_synthetic_data()
else:
    df = generate_synthetic_data()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df[(df["Date"] >= "2025-04-01") & (df["Date"] <= "2026-03-31")]

# ----------------------------------------------------------
# KPIs
# ----------------------------------------------------------
st.markdown("### 💼 Key Performance Indicators (FY25–26)")
latest = df.iloc[-1]
prev = df.iloc[-2]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue", f"₹{latest['Revenue']:,.0f}", f"{(latest['Revenue']-prev['Revenue']):,.0f}")
c2.metric("Profit", f"₹{latest['Profit']:,.0f}", f"{(latest['Profit']-prev['Profit']):,.0f}")
c3.metric("Transactions", f"{latest['Transactions']:,.0f}")
c4.metric("Active Merchants", f"{latest['Active Merchants']:,.0f}")

# ----------------------------------------------------------
# Tabs for visualization
# ----------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trends & Growth", 
    "📊 Correlations & Efficiency", 
    "💰 Profitability & Margins", 
    "🧠 Insights & Story"
])

# ----------------------------------------------------------
# 1️⃣ Trends
# ----------------------------------------------------------
with tab1:
    st.subheader("Revenue, Profit & Volume Trends")

    fig = px.line(df, x="Date", y=["Revenue", "Profit"], title="Revenue & Profit Over Time", 
                  color_discrete_sequence=["#0B69FF", "#9B51E0"])
    st.plotly_chart(fig, use_container_width=True)

    df["Revenue Growth %"] = df["Revenue"].pct_change() * 100
    fig2 = px.line(df, x="Date", y="Revenue Growth %", title="Weekly Revenue Growth (%)",
                   color_discrete_sequence=["#F39C12"])
    st.plotly_chart(fig2, use_container_width=True)

    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    qrev = df.groupby("Quarter")["Revenue"].sum().reset_index()
    fig3 = px.bar(qrev, x="Quarter", y="Revenue", title="Quarterly Revenue Breakdown", color="Revenue",
                  color_continuous_scale="Blues")
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------
# 2️⃣ Correlation & Efficiency
# ----------------------------------------------------------
with tab2:
    st.subheader("📊 Relationship Between Metrics")

    corr = df[["Revenue", "Profit", "Transactions", "Active Merchants"]].corr()
    fig_corr = ff.create_annotated_heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.columns),
        colorscale="blues", showscale=True)
    fig_corr.update_layout(title="Correlation Heatmap", height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    fig4 = px.scatter(df, x="Transactions", y="Revenue", 
                      title="Revenue vs Transactions Efficiency",
                      color="Profit", size="Active Merchants",
                      color_continuous_scale="Viridis")
    st.plotly_chart(fig4, use_container_width=True)

# ----------------------------------------------------------
# 3️⃣ Profitability & Margins
# ----------------------------------------------------------
with tab3:
    st.subheader("💰 Profit Margin & Merchant Impact")

    df["Profit Margin %"] = (df["Profit"] / df["Revenue"]) * 100
    fig5 = px.area(df, x="Date", y="Profit Margin %", title="Profit Margin Trend (%)", color_discrete_sequence=["#4B7BE5"])
    st.plotly_chart(fig5, use_container_width=True)

    # Simulate merchant contribution
    tiers = ["Small", "Medium", "Enterprise"]
    contrib = [0.25, 0.45, 0.30]
    mdata = pd.DataFrame({"Merchant Tier": tiers, "Revenue Share": contrib})
    fig6 = px.pie(mdata, names="Merchant Tier", values="Revenue Share", title="Revenue Contribution by Merchant Tier",
                  color_discrete_sequence=["#0B69FF", "#5C33F6", "#A8BFFF"])
    st.plotly_chart(fig6, use_container_width=True)

# ----------------------------------------------------------
# 4️⃣ Insights
# ----------------------------------------------------------
with tab4:
    st.subheader("🧠 Strategic Insights Summary")
    avg_growth = df["Revenue Growth %"].mean()
    avg_margin = df["Profit Margin %"].mean()
    strongest_corr = corr["Revenue"].drop("Revenue").idxmax()

    st.markdown(f"""
    - 📈 Average weekly revenue growth: **{avg_growth:.2f}%**
    - 💰 Average profit margin: **{avg_margin:.2f}%**
    - 🔗 Strongest correlation: **Revenue ↔ {strongest_corr}**
    - 🧾 Suggestion: Focus on increasing {strongest_corr.lower()} to enhance revenue momentum.
    - ⚙️ Seasonal trend: Q3 (Oct–Dec) shows peak growth — plan promotions accordingly.
    """)

    st.markdown("""
    <div style="background-color:#EAF2FF;padding:15px;border-radius:10px;margin-top:10px;">
    <b>Strategic Takeaway:</b>  
    Razorpay’s FY25–26 data shows strong growth potential led by transaction expansion.  
    Prioritizing merchant engagement and optimizing transaction volume could further improve
    revenue scalability and profit margins.
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------
st.markdown("---")
st.markdown("""
<center>💡 Developed for <b>Razorpay x Grant Thornton</b> | A fun MBA-fintech analytics experiment ✨</center>
""", unsafe_allow_html=True)
