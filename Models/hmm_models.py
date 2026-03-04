import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

class MarketHMM:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.hmm_model = None
        self.scaler = StandardScaler()

    def train_and_predict(self, data):
        print(f"Training HMM model with {self.n_regimes} regimes...")
        
        X = data[['Returns', 'Volatility', 'Momentum', 'Liquidity']].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.hmm_model = GaussianHMM(n_components=self.n_regimes, covariance_type='full', n_iter=1000, random_state=42)
        self.hmm_model.fit(X_scaled)
        
        states = self.hmm_model.predict(X_scaled)
        
        data['Regime'] = states
        
        print("Model trained and regimes assigned!")
        return data