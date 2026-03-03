import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class RegimeDetector:
    def __init__(self,ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
    def fetch_data(self):
        print(f"Fetching data for {self.ticker}...")
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        self.data = df
        price_col = 'Adj Close' if 'Adj Close' in self.data.columns else 'Close'
        self.returns = np.log(self.data[price_col] / self.data[price_col].shift(1)).dropna()
        
        print(f"Data fetched successfully! (Using column: {price_col})")
    def compute_features(self):
        self.data['Returns'] = self.returns
        self.data['Volatility'] = self.returns.rolling(window=20).std()
        self.data['Momentum'] = self.returns.rolling(window=20).mean()
        self.data['Liquidity'] = np.log(self.data['Volume'] / self.data['Volume'].shift(1))
        
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.dropna(inplace=True)
        self.returns = self.returns.loc[self.data.index]
        
        print("Features computed: Returns, Volatility, Momentum, Liquidity")
        
    def train_hmm(self, n_regimes=3):
        X= self.data[['Returns','Volatility','Momentum','Liquidity']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.hmm_model = GaussianHMM(n_components=n_regimes, covariance_type='full', n_iter=1000, random_state=42)
        self.hmm_model.fit(X_scaled)
        states = self.hmm_model.predict(X_scaled)
        self.data['Regime'] = states
        print(f"HMM trained with {n_regimes} states. Regimes assigned to data.")

    def plot_regimes(self):
            plt.figure(figsize=(14, 7))
            
            plt.plot(self.data.index, self.data['Close'], color='black', alpha=0.3, lw=0.5)
            
            for regime in self.data['Regime'].unique():
                regime_data = self.data[self.data['Regime'] == regime]
                plt.scatter(regime_data.index, regime_data['Close'], label=f'Regime {regime}', s=10)
                
            plt.title(f'Market Regimes Detection for {self.ticker}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.show()
        
if __name__ == "__main__":
    detector = RegimeDetector(ticker="^GSPC", start_date="2005-01-01", end_date="2024-01-01")
    detector.fetch_data()
    detector.compute_features()
    
    detector.train_hmm(n_regimes=3)
    
    print(detector.data[['Close', 'Volatility', 'Momentum', 'Liquidity', 'Regime']].tail(10))
    detector.plot_regimes()