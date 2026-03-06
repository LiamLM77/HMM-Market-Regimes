# HMM Market Regimes

Hidden Markov Model (HMM)-based market regime detection and dynamic exposure strategy, delivered as an interactive Streamlit dashboard.

This project trains a Gaussian HMM on engineered market features, estimates regime probabilities over time, and converts those probabilities into a continuous long exposure signal for out-of-sample backtesting.

## Features

- Regime detection with `hmmlearn` Gaussian HMM
- Feature engineering from market data (`Returns`, `Volatility`, `Momentum`, `Liquidity`)
- Train/test split with explicit out-of-sample evaluation date
- Continuous allocation logic: lower exposure when crisis-regime probability increases
- Transaction cost-aware strategy returns
- Interactive Plotly visualizations (equity curves, exposure, transition matrix heatmap, and 3D feature clustering)

## Project Structure

```text
HMM-Market-Regimes/
|-- main.py                    # Streamlit app entry point
|-- requirements.txt
|-- Data/
|   `-- data_loader.py         # Download data + feature engineering
|-- Models/
|   `-- hmm_models.py          # HMM training, prediction, transition matrix
`-- Backtest/
    `-- strategy.py            # Position sizing, returns, KPIs
```

## How It Works

1. Download historical data from Yahoo Finance using `yfinance`.
2. Compute log returns and rolling features: `Volatility` (20-day rolling std), `Momentum` (20-day rolling mean), and `Liquidity` (log change in volume).
3. Fit a `GaussianHMM` on in-sample data only.
4. Predict regimes and regime probabilities over the full timeline.
5. Define the crisis regime as the regime with highest average in-sample volatility.
6. Build exposure as:

```text
Position = 1 - P(crisis regime)
```

7. Backtest strategy returns with transaction costs and compare against market buy-and-hold.

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The App

From the repository root:

```bash
streamlit run main.py
```

Then open the local Streamlit URL shown in your terminal.

## Using The Dashboard

Set parameters in the sidebar:

- `Ticker` (default: `^GSPC`)
- `Start Date`
- `Train End Date (Split)`
- `Number of Regimes` (2 to 5)

Click **Run Advanced Backtest** to:

- Train the model in-sample
- Predict regimes for the full dataset
- Evaluate out-of-sample KPIs
- Render all interactive charts

## Output Metrics

The app reports out-of-sample:

- Annualized market return
- Annualized strategy return
- Market Sharpe ratio
- Strategy Sharpe ratio
- Market max drawdown
- Strategy max drawdown
- Total trades executed

## Notes And Assumptions

- Data source is Yahoo Finance; availability and adjusted fields depend on ticker.
- The strategy is long-only with dynamic exposure in `[0, 1]`.
- Regime labels are latent and do not have fixed economic meaning across retrains.
- This is a research/educational framework, not financial advice.

## Potential Extensions

- Walk-forward retraining
- Regime-specific transaction costs/slippage modeling
- Alternative feature sets and window lengths
- Multi-asset portfolio construction
- Formal statistical tests for regime persistence and stability

## License

See `LICENSE`.
