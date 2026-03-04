import yfinance as yf
import pandas as pd
import numpy as np

class MarketDataLoader:
    def __init__(self, ticker, start_date, end_date):
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
        return self.data

    def compute_features(self):
        self.data['Returns'] = self.returns
        self.data['Volatility'] = self.returns.rolling(window=20).std()
        self.data['Momentum'] = self.returns.rolling(window=20).mean()
        self.data['Liquidity'] = np.log(self.data['Volume'] / self.data['Volume'].shift(1))
        
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.dropna(inplace=True)
        self.returns = self.returns.loc[self.data.index]
        
        print("Features computed: Returns, Volatility, Momentum, Liquidity")
        return self.data