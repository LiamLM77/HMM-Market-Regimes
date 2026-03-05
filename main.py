import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Data.data_loader import MarketDataLoader
from Models.hmm_models import MarketHMM
from Backtest.strategy import HMMStrategy

st.set_page_config(page_title="HMM Quant Strategy", layout="wide")

st.title("Market Regime Detection & Backtest")

ticker = st.sidebar.text_input("Ticker", "^GSPC")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2005-01-01"))
n_regimes = st.sidebar.slider("Number of Regimes", 2, 5, 3)
df = None

if st.sidebar.button("Run Backtest"):
    with st.spinner("Fetching data and training HMM..."):
        loader = MarketDataLoader(ticker, start_date.strftime("%Y-%m-%d"), "2024-01-01")
        df = loader.fetch_data()
        df = loader.compute_features()
        
        model = MarketHMM(n_regimes=n_regimes)
        df = model.train_and_predict(df)
        
        strategy = HMMStrategy(transaction_cost=0.001)
        df = strategy.run_backtest(df)
        kpis = strategy.calculate_kpis(df)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strategy Return (Ann)", kpis["Strategy Return (Ann)"], kpis["Market Return (Ann)"])
        col2.metric("Strategy Sharpe", kpis["Strategy Sharpe"], kpis["Market Sharpe"])
        col3.metric("Strategy Max Drawdown", kpis["Strategy Max Drawdown"], kpis["Market Max Drawdown"])
        col4.metric("Trades Executed", int(df['Trades'].sum()))

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Market'], line=dict(color='gray'), name='Market'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Strategy'], line=dict(color='blue'), name='Strategy'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Regime'], mode='markers', marker=dict(size=4, color=df['Regime']), name='Regime'), row=2, col=1)
        
        fig.update_layout(height=700, template="plotly_dark", title_text="Backtest Results & Market Regimes")
        st.plotly_chart(fig, use_container_width=True)
        
st.markdown("---")
st.subheader("3D Feature Space Clustering")
if df is not None:
    fig3d = go.Figure(data=[go.Scatter3d(
            x=df['Volatility'],
            y=df['Momentum'],
            z=df['Liquidity'],
            mode='markers',
            marker=dict(
                size=3,
                color=df['Regime'],
                colorscale='Portland',
                opacity=0.7
            ),
            text=df.index.strftime('%Y-%m-%d')
            )])
            
    fig3d.update_layout(
            scene=dict(
                xaxis_title='Volatility',
                yaxis_title='Momentum',
                zaxis_title='Liquidity',
                bgcolor='rgb(20, 20, 20)'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            template="plotly_dark",
            height=600
            )
            
    st.plotly_chart(fig3d, use_container_width=True)
else:
    st.info("Run a backtest to view the 3D feature clustering plot.")