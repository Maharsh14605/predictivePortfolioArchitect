# Predictive Portfolio Architect

Predictive Portfolio Architect is a Streamlit-based financial analytics application that uses daily market data, machine learning, and risk-based portfolio allocation logic to recommend investment weights across a user-selected list of stocks.

## Features

- Downloads historical daily stock data using yfinance
- Stores market data in a local SQLite database
- Calculates daily returns, moving averages, rolling volatility, and momentum-style features
- Trains Random Forest models to predict next-day stock returns
- Recommends portfolio weights based on predicted return, volatility, and user risk tolerance
- Calculates expected annual return, annual volatility, and Sharpe Ratio
- Compares AI portfolio performance against a benchmark such as the S&P 500
- Includes an automated GitHub Actions workflow for daily data updates and model retraining

## Tech Stack

- Python
- pandas
- NumPy
- yfinance
- scikit-learn
- SQLite
- Streamlit
- matplotlib
- joblib
- GitHub Actions

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python updateDaily.py
streamlit run app.py