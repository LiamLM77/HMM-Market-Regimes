import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

class MarketHMM:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.hmm_model = None
        self.scaler = StandardScaler()

    def fit(self, train_data):
        X_train = train_data[['Returns', 'Volatility', 'Momentum', 'Liquidity']].values
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.hmm_model = GaussianHMM(n_components=self.n_regimes, covariance_type='full', n_iter=1000, random_state=42)
        self.hmm_model.fit(X_scaled)

    def predict(self, data):
        X = data[['Returns', 'Volatility', 'Momentum', 'Liquidity']].values
        X_scaled = self.scaler.transform(X)
        
        data['Regime'] = self.hmm_model.predict(X_scaled)
        probs = self.hmm_model.predict_proba(X_scaled)
        
        for i in range(self.n_regimes):
            data[f'Prob_Regime_{i}'] = probs[:, i]
            
        return data

    def get_transition_matrix(self):
        return self.hmm_model.transmat_