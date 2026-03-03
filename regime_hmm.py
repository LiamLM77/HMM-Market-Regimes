import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

class RegimeDetector:
    def __init__(self,ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
    def fetch_data(self):
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.returns = np.log(self.data['Adj Close']/self.data['Adj Close'].shift(1)).dropna()
        print(f"Data fetched for {self.ticker} from {self.start_date} to {self.end_date}")
    def compute_features(self):
        self.data['Returns'] = self.returns
        self.data['Volatility'] = self.returns.rolling(window=20).std()
        self.data['Momentum'] = self.returns.rolling(window=20).mean()
        self.data['Liquidity'] = np.log(self.data['Volume'] / self.data['Volume'].shift(1))
        self.data.dropna(inplace=True)
        self.returns=self.returns.loc[self.data.index]
        print("Features computed: Returns, Volatility, Momentum,Liquidity")
        
    def train_hmm(self,n_states=3):
        X= self.data[['Returns','Volatility','Momentum','Liquidity']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.hmm_model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=1000, randiom_state=42)
        self.hmm_model.fit(X_scaled)
        states = self.hmm_model.predict(X_scaled)
        self.data['Regime'] = states
        print(f"HMM trained with {n_states} states. Regimes assigned to data.")

if __name__ == "__main__":
    detector = RegimeDetector(ticker="^GSPC", start_date="2005-01-01", end_date="2024-01-01")
    detector.fetch_data()
    detector.compute_features()
    
    # On lance l'entraînement !
    detector.train_hmm(n_regimes=3)
    
    # On regarde les 10 dernières lignes pour voir si la colonne 'Regime' s'affiche bien (avec des 0, 1 ou 2)
    print(detector.data[['Adj Close', 'Volatility', 'Momentum', 'Liquidity', 'Regime']].tail(10))