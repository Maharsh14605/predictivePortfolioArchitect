import yfinance as yf
import pandas as pd


def downloadPrices(tickers, period="5y"):
    stockData = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if stockData.empty:
        raise ValueError("No data was downloaded. Check ticker symbols or internet connection.")

    if isinstance(stockData.columns, pd.MultiIndex):
        prices = stockData["Close"]
    else:
        prices = stockData[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all")
    prices = prices.ffill()

    return prices


def calculateDailyReturns(prices):
    dailyReturns = prices.pct_change()
    dailyReturns = dailyReturns.dropna()

    return dailyReturns