# =============================
# Libraries / Dependencies
# =============================
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from Data.data_loader import MarketDataLoader
from Models.hmm_models import MarketHMM
from Backtest.strategy import HMMStrategy

# =============================
# App Setup
# =============================
# Streamlit app configuration and title.
st.set_page_config(page_title="HMM Quant Strategy", layout="wide")
st.title("Advanced Market Regime HMM (OOS & Continuous Allocation)")

# =============================
# Sidebar Inputs
# =============================
# Sidebar controls for instrument, dates, and model complexity.
ticker = st.sidebar.text_input("Ticker", "^GSPC")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2005-01-01"))
train_end_date = st.sidebar.date_input("Train End Date (Split)", value=pd.to_datetime("2015-01-01"))
n_regimes = st.sidebar.slider("Number of Regimes", 2, 5, 3)

# =============================
# Backtest Execution Pipeline
# =============================
if st.sidebar.button("Run Advanced Backtest"):
    with st.spinner("Training on In-Sample, Predicting Out-Of-Sample..."):
        # 1) Load historical data and engineer features.
        loader = MarketDataLoader(ticker, start_date.strftime("%Y-%m-%d"), "2024-01-01")
        df = loader.fetch_data()
        df = loader.compute_features()
        
        # 2) Split by date: train on history up to the split.
        train_df = df[:train_end_date.strftime("%Y-%m-%d")]
        
        # 3) Fit HMM regimes and infer regime probabilities on full data.
        model = MarketHMM(n_regimes=n_regimes)
        model.fit(train_df)
        df = model.predict(df)
        transmat = model.get_transition_matrix()
        
        # 4) Convert probabilities into dynamic portfolio allocation.
        strategy = HMMStrategy(transaction_cost=0.001)
        df = strategy.run_backtest(df, train_end_date.strftime("%Y-%m-%d"))
        
        # 5) Report out-of-sample KPIs after the training split.
        df_oos = df[train_end_date.strftime("%Y-%m-%d"):]
        kpis_oos = strategy.calculate_kpis(df_oos)
        
        # =============================
        # KPI Summary (Out-of-Sample)
        # =============================
        st.subheader("Out-Of-Sample Performance (After Training Date)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("OOS Strategy Return", kpis_oos["Strategy Return (Ann)"], kpis_oos["Market Return (Ann)"])
        col2.metric("OOS Strategy Sharpe", kpis_oos["Strategy Sharpe"], kpis_oos["Market Sharpe"])
        col3.metric("OOS Strategy Drawdown", kpis_oos["Strategy Max Drawdown"], kpis_oos["Market Max Drawdown"])
        col4.metric("Total Trades Executed", f"{df['Trades'].sum():.1f}")

        # =============================
        # Main Backtest Chart
        # =============================
        # Top panel: equity curves. Bottom panel: dynamic exposure.
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Market'], line=dict(color='gray'), name='Market'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Strategy'], line=dict(color='blue'), name='Strategy'), row=1, col=1)
        
        # Mark the train/test split for visual clarity.
        fig.add_vline(x=pd.to_datetime(train_end_date), line_width=2, line_dash="dash", line_color="red", row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Position'], fill='tozeroy', line=dict(color='cyan'), name='Exposure %'), row=2, col=1)
        
        fig.update_layout(height=700, template="plotly_dark", title_text="Continuous Allocation Backtest")
        st.plotly_chart(fig, use_container_width=True)
        
        # =============================
        # Diagnostics Charts
        # =============================
        st.markdown("---")
        col_heat, col_3d = st.columns(2)
        
        with col_heat:
            st.subheader("Transition Matrix Heatmap")
            fig_heat = px.imshow(transmat, text_auto=".2%", color_continuous_scale="Viridis", labels=dict(x="To Regime", y="From Regime", color="Probability"))
            fig_heat.update_layout(template="plotly_dark")
            st.plotly_chart(fig_heat, use_container_width=True)
            
        with col_3d:
            st.subheader("3D Feature Space Clustering")
            fig3d = go.Figure(data=[go.Scatter3d(x=df['Volatility'], y=df['Momentum'], z=df['Liquidity'], mode='markers', marker=dict(size=3, color=df['Regime'], colorscale='Portland', opacity=0.7), text=df.index.strftime('%Y-%m-%d'))])
            fig3d.update_layout(scene=dict(xaxis_title='Volatility', yaxis_title='Momentum', zaxis_title='Liquidity', bgcolor='rgb(20, 20, 20)'), margin=dict(l=0, r=0, b=0, t=0), template="plotly_dark", height=400)
            st.plotly_chart(fig3d, use_container_width=True)