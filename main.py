import plotly.graph_objects as go
from data.data_loader import MarketDataLoader
from models.hmm_model import MarketHMM
from backtest.strategy import HMMStrategy

def plot_interactive_results(data, ticker):
    import plotly.graph_objects as go
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Cumulative_Market'],
        mode='lines', 
        name='Buy & Hold', 
        line=dict(color='gray', width=1.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Cumulative_Strategy'],
        mode='lines', 
        name='HMM Strategy', 
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=f"HMM Strategy vs Market ({ticker})",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (1$ = Start)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    fig.show()

if __name__ == "__main__":
    ticker = "^GSPC"
    
    loader = MarketDataLoader(ticker, "2005-01-01", "2026-01-01")
    df = loader.fetch_data()
    df = loader.compute_features()
    
    model = MarketHMM(n_regimes=3)
    df = model.train_and_predict(df)
    
    strategy = HMMStrategy()
    df = strategy.run_backtest(df)
    
    plot_interactive_results(df, ticker)