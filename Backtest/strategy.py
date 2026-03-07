# =============================
# Libraries / Dependencies
# =============================
import numpy as np
import pandas as pd

# =============================
# Strategy Logic
# =============================
class HMMStrategy:
    """Continuous-allocation strategy driven by HMM crisis probability."""

    def __init__(self, transaction_cost=0.001):
        """Initialize strategy parameters."""
        self.crisis_regime = None
        self.tc = transaction_cost

    def run_backtest(self, data, train_end_date):
        """Run out-of-sample style backtest using regime probabilities as exposure."""
        # =============================
        # 1) Identify Crisis Regime
        # =============================
        train_data = data[:train_end_date]
        # Define the "crisis" regime as the most volatile regime in training data.
        regime_vols = train_data.groupby('Regime')['Volatility'].mean()
        self.crisis_regime = regime_vols.idxmax()
        
        crisis_prob_col = f'Prob_Regime_{self.crisis_regime}'
        
        # =============================
        # 2) Build Position and Returns
        # =============================
        # Less crisis probability => higher market exposure (0 to 1 allocation).
        data['Position'] = 1.0 - data[crisis_prob_col]
        
        data['Trades'] = data['Position'].diff().abs().fillna(0)
        # Use previous day's position to avoid look-ahead bias.
        data['Strategy_Returns'] = (data['Position'].shift(1) * data['Returns']) - (data['Trades'] * self.tc)
        data.dropna(inplace=True)
        
        data['Cumulative_Market'] = np.exp(data['Returns'].cumsum())
        data['Cumulative_Strategy'] = np.exp(data['Strategy_Returns'].cumsum())
        
        return data

    def calculate_kpis(self, data):
        """Compute annualized return, Sharpe ratio, and max drawdown metrics."""
        # =============================
        # 3) Performance Metrics
        # =============================
        days = len(data)
        ann_ret_mkt = (data['Cumulative_Market'].iloc[-1]) ** (252/days) - 1
        ann_ret_strat = (data['Cumulative_Strategy'].iloc[-1]) ** (252/days) - 1
        
        sharpe_mkt = np.sqrt(252) * data['Returns'].mean() / data['Returns'].std()
        sharpe_strat = np.sqrt(252) * data['Strategy_Returns'].mean() / data['Strategy_Returns'].std()
        
        roll_max_strat = data['Cumulative_Strategy'].cummax()
        mdd_strat = (data['Cumulative_Strategy'] / roll_max_strat - 1).min()
        
        roll_max_mkt = data['Cumulative_Market'].cummax()
        mdd_mkt = (data['Cumulative_Market'] / roll_max_mkt - 1).min()

        return {
            "Market Return (Ann)": f"{ann_ret_mkt:.2%}",
            "Strategy Return (Ann)": f"{ann_ret_strat:.2%}",
            "Market Sharpe": f"{sharpe_mkt:.2f}",
            "Strategy Sharpe": f"{sharpe_strat:.2f}",
            "Market Max Drawdown": f"{mdd_mkt:.2%}",
            "Strategy Max Drawdown": f"{mdd_strat:.2%}"
        }