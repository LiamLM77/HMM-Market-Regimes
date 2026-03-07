# =============================
# Libraries / Dependencies
# =============================
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# =============================
# HMM Model Wrapper
# =============================
class MarketHMM:
    """Wrapper around GaussianHMM for market regime detection."""

    def __init__(self, n_regimes=3):
        """Set number of hidden regimes and scaler used for feature normalization."""
        self.n_regimes = n_regimes
        self.hmm_model = None
        self.scaler = StandardScaler()

    def fit(self, train_data):
        """Fit the HMM on in-sample data after feature scaling."""
        # =============================
        # 1) Prepare Training Features
        # =============================
        X_train = train_data[['Returns', 'Volatility', 'Momentum', 'Liquidity']].values
        X_scaled = self.scaler.fit_transform(X_train)
        
        # =============================
        # 2) Train Gaussian HMM
        # =============================
        self.hmm_model = GaussianHMM(n_components=self.n_regimes, covariance_type='full', n_iter=1000, random_state=42)
        self.hmm_model.fit(X_scaled)

    def predict(self, data):
        """Infer regime labels and regime probabilities for each row."""
        # =============================
        # 3) Regime Inference
        # =============================
        X = data[['Returns', 'Volatility', 'Momentum', 'Liquidity']].values
        X_scaled = self.scaler.transform(X)
        
        data['Regime'] = self.hmm_model.predict(X_scaled)
        probs = self.hmm_model.predict_proba(X_scaled)
        
        # Store each regime probability as a separate column for strategy logic.
        for i in range(self.n_regimes):
            data[f'Prob_Regime_{i}'] = probs[:, i]
            
        return data

    def get_transition_matrix(self):
        """Return the trained transition matrix between hidden regimes."""
        return self.hmm_model.transmat_