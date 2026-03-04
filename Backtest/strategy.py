import numpy as np
import pandas as pd

class HMMStrategy:
    def __init__(self):
        self.crisis_regime = None

    def run_backtest(self, data):
        regime_vols = data.groupby('Regime')['Volatility'].mean()
        self.crisis_regime = regime_vols.idxmax()
        
        data['Position'] = 1
        data.loc[data['Regime'] == self.crisis_regime, 'Position'] = 0
        
        data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
        data.dropna(inplace=True)
        
        data['Cumulative_Market'] = np.exp(data['Returns'].cumsum())
        data['Cumulative_Strategy'] = np.exp(data['Strategy_Returns'].cumsum())
        
        return data