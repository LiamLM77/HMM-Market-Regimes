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

        # Re-label regimes so lower IDs always mean calmer (lower volatility) markets.
        split_key = train_end_date.strftime("%Y-%m-%d")
        train_labeled_raw = df[:split_key]
        old_order = (
            train_labeled_raw.groupby('Regime')['Volatility']
            .mean()
            .sort_values()
            .index
            .astype(int)
            .tolist()
        )
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(old_order)}

        # Remap regime labels.
        df['Regime'] = df['Regime'].map(old_to_new).astype(int)

        # Remap probability columns to the new regime IDs.
        prob_by_new_id = {
            new_id: df[f'Prob_Regime_{old_id}'].copy()
            for old_id, new_id in old_to_new.items()
        }
        for new_id in range(n_regimes):
            df[f'Prob_Regime_{new_id}'] = prob_by_new_id[new_id]

        # Reorder transition matrix rows/cols to the new regime ordering.
        transmat = transmat[old_order][:, old_order]
        
        # 4) Convert probabilities into dynamic portfolio allocation.
        strategy = HMMStrategy(transaction_cost=0.001)
        df = strategy.run_backtest(df, train_end_date.strftime("%Y-%m-%d"))
        
        # 5) Report out-of-sample KPIs after the training split.
        df_oos = df[train_end_date.strftime("%Y-%m-%d"):]
        kpis_oos = strategy.calculate_kpis(df_oos)

        # Build logical regime colors from calm -> crisis using volatility rank.
        train_labeled = df[:train_end_date.strftime("%Y-%m-%d")]
        crisis_regime = int(strategy.crisis_regime)
        vol_by_regime = train_labeled.groupby('Regime')['Volatility'].mean().sort_values()
        risk_order = [int(r) for r in vol_by_regime.index]

        # Low risk (green) -> mid risk (yellow/orange) -> high risk (red).
        risk_palette = ['#2E7D32', '#66BB6A', '#FBC02D', '#FB8C00', '#C62828']
        regime_color_map = {}
        n_risks = len(risk_order)
        for idx, regime_int in enumerate(risk_order):
            if n_risks == 1:
                palette_idx = len(risk_palette) - 1
            else:
                palette_idx = round(idx * (len(risk_palette) - 1) / (n_risks - 1))
            regime_color_map[regime_int] = risk_palette[palette_idx]

        # Always force the strategy crisis regime to red for intuitive reading.
        regime_color_map[crisis_regime] = '#C62828'

        regime_name_map = {}
        for idx, regime_int in enumerate(risk_order):
            if regime_int == crisis_regime:
                regime_name_map[regime_int] = 'Crisis / High Volatility'
            elif idx == 0:
                regime_name_map[regime_int] = 'Calm / Low Volatility'
            elif idx <= (n_risks - 1) / 2:
                regime_name_map[regime_int] = 'Neutral / Moderate Volatility'
            else:
                regime_name_map[regime_int] = 'Elevated Risk / Higher Volatility'
        
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
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Cumulative_Strategy'],
                mode='lines',
                line=dict(color='#1E3A8A', width=2.2),
                name='Strategy',
                showlegend=True
            ),
            row=1,
            col=1
        )

        # Clean regime overlay: line segments are colored by active regime (no dot clutter).
        for regime_int in sorted(regime_color_map.keys()):
            y_regime = df['Cumulative_Strategy'].where(df['Regime'] == regime_int)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=y_regime,
                    mode='lines',
                    line=dict(color=regime_color_map[regime_int], width=2.6),
                    name=f"Regime {regime_int}: {regime_name_map.get(regime_int, 'Unlabeled')}"
                ),
                row=1,
                col=1
            )
        
        # Mark the train/test split for visual clarity.
        fig.add_vline(x=pd.to_datetime(train_end_date), line_width=2, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_annotation(
            x=pd.to_datetime(train_end_date),
            y=1.02,
            yref="paper",
            text="Train/Test Split",
            showarrow=False,
            font=dict(color="red", size=11)
        )
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Position'], fill='tozeroy', line=dict(color='cyan'), name='Exposure %'), row=2, col=1)

        fig.update_yaxes(title_text="Growth of $1", row=1, col=1)
        fig.update_yaxes(title_text="Exposure", tickformat=".0%", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_layout(
            height=700,
            template="plotly_white",
            title_text="Continuous Allocation Backtest",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # =============================
        # Diagnostics Charts
        # =============================
        st.markdown("---")
        col_heat, col_3d = st.columns(2)
        
        with col_heat:
            st.subheader("Transition Matrix Heatmap")
            # Focus color range on rare transition moves (1% is already meaningful).
            heat_max = 0.05  # 5%
            fig_heat = px.imshow(
                transmat,
                text_auto=".2%",
                color_continuous_scale="YlOrRd",
                zmin=0,
                zmax=heat_max,
                labels=dict(x="To Regime", y="From Regime", color="Probability")
            )
            fig_heat.update_layout(
                template="plotly_white",
                margin=dict(l=10, r=10, t=40, b=10),
                coloraxis_colorbar=dict(
                    title="Probability",
                    tickformat=".1%",
                    tickvals=[0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
                )
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption("Color scale is clipped to 0%-5% so low-probability regime changes are easier to read.")
            
        with col_3d:
            st.subheader("3D Feature Space Clustering")
            fig3d = go.Figure()
            fig3d.add_trace(
                go.Scatter3d(
                    x=df['Volatility'],
                    y=df['Momentum'],
                    z=df['Liquidity'],
                    mode='markers',
                    showlegend=False,
                    marker=dict(size=3, color=df['Regime'].map(regime_color_map), opacity=0.7),
                    text=df.index.strftime('%Y-%m-%d')
                )
            )

            # Add legend-only traces so users can map each regime to its color.
            unique_regimes = sorted(df['Regime'].dropna().unique())
            for regime in unique_regimes:
                regime_int = int(regime)
                regime_color = regime_color_map.get(regime_int, '#9E9E9E')
                fig3d.add_trace(
                    go.Scatter3d(
                        x=[None],
                        y=[None],
                        z=[None],
                        mode='markers',
                        name=f"Regime {regime_int}: {regime_name_map.get(regime_int, 'Unlabeled')}",
                        marker=dict(size=6, color=regime_color),
                        showlegend=True
                    )
                )

            fig3d.update_layout(
                scene=dict(
                    xaxis_title='Volatility',
                    yaxis_title='Momentum',
                    zaxis_title='Liquidity',
                    bgcolor='rgb(18, 22, 28)'
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                template="plotly_dark",
                height=400,
                legend=dict(title='Regimes')
            )
            st.plotly_chart(fig3d, use_container_width=True)