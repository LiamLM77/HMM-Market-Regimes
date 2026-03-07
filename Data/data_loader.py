# =============================
# Libraries / Dependencies
# =============================
import yfinance as yf
import pandas as pd
import numpy as np

# =============================
# Data Loader
# =============================
class MarketDataLoader:
    """Download market data and build model-ready feature columns."""

    def __init__(self, ticker, start_date, end_date):
        """Store request parameters and internal data containers."""
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None

    def fetch_data(self):
        """Download OHLCV data and compute log returns from adjusted/close price."""
        # =============================
        # 1) Download Raw Market Data
        # =============================
        print(f"Fetching data for {self.ticker}...")
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance can return a multi-index for some tickers; flatten it.
            df.columns = df.columns.droplevel(1)
            
        self.data = df
        # Prefer adjusted close when available to account for corporate actions.
        price_col = 'Adj Close' if 'Adj Close' in self.data.columns else 'Close'
        self.returns = np.log(self.data[price_col] / self.data[price_col].shift(1)).dropna()
        
        print(f"Data fetched successfully! (Using column: {price_col})")
        return self.data

    def compute_features(self):
        """Create features used by the HMM and align/clean the resulting dataset."""
        # =============================
        # 2) Feature Engineering
        # =============================
        self.data['Returns'] = self.returns
        self.data['Volatility'] = self.returns.rolling(window=20).std()
        self.data['Momentum'] = self.returns.rolling(window=20).mean()
        self.data['Liquidity'] = np.log(self.data['Volume'] / self.data['Volume'].shift(1))
        
        # Remove infinities/NaNs from rolling and ratio operations.
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.dropna(inplace=True)
        # Keep returns indexed exactly like the cleaned feature frame.
        self.returns = self.returns.loc[self.data.index]
        
        print("Features computed: Returns, Volatility, Momentum, Liquidity")
        return self.data