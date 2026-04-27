import streamlit as st
import pandas as pd

from src.backtest import compareStrategyToBenchmark
from src.dataLoader import downloadPrices, calculateDailyReturns
from src.database import loadPricesFromDatabase, savePricesToDatabase
from src.features import createFeatureData
from src.model import trainModelForStock, saveModel
from src.portfolio import calculateWeights, calculatePortfolioMetrics


st.set_page_config(
    page_title="Predictive Portfolio Architect",
    page_icon="📈",
    layout="wide"
)


st.title("Predictive Portfolio Architect")

st.write(
    "A machine learning portfolio tool that uses daily market data, "
    "predictive modeling, risk tolerance, and Sharpe Ratio analysis "
    "to recommend portfolio allocations."
)

st.caption("Educational project only. This is not financial advice.")


@st.cache_data(ttl=86400)
def loadData(tickers):
    prices = loadPricesFromDatabase(tickers)

    missingData = prices.empty or any(ticker not in prices.columns for ticker in tickers)

    if missingData:
        prices = downloadPrices(tickers, period="5y")
        savePricesToDatabase(prices)

    dailyReturns = calculateDailyReturns(prices)

    return prices, dailyReturns


tickerInput = st.sidebar.text_input(
    "Enter stock tickers separated by commas",
    value="RY.TO, TD.TO, BNS.TO, AAPL, MSFT, AMZN"
)

riskTolerance = st.sidebar.slider(
    "Risk Tolerance",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1
)

investmentAmount = st.sidebar.number_input(
    "Investment Amount",
    min_value=1000,
    value=10000,
    step=1000
)

benchmarkTicker = st.sidebar.selectbox(
    "Benchmark",
    options=["^GSPC", "^GSPTSE"],
    index=0
)

runButton = st.sidebar.button("Build Portfolio")

tickers = [
    ticker.strip().upper()
    for ticker in tickerInput.split(",")
    if ticker.strip()
]

if not runButton:
    st.info("Enter tickers, choose your risk tolerance, and click Build Portfolio.")
    st.stop()

if len(tickers) < 2:
    st.warning("Please enter at least two tickers.")
    st.stop()

try:
    prices, dailyReturns = loadData(tickers)
    featureData = createFeatureData(prices)

    expectedReturns = {}
    modelMetrics = {}

    for ticker, stockData in featureData.items():
        model, expectedReturn, metrics = trainModelForStock(stockData)

        saveModel(model, ticker)

        expectedReturns[ticker] = expectedReturn
        modelMetrics[ticker] = metrics

    weights = calculateWeights(
        expectedReturns=expectedReturns,
        dailyReturns=dailyReturns,
        riskTolerance=riskTolerance
    )

    portfolioMetrics = calculatePortfolioMetrics(
        weights=weights,
        dailyReturns=dailyReturns
    )

    st.subheader("Portfolio Summary")

    column1, column2, column3 = st.columns(3)

    column1.metric(
        "Expected Annual Return",
        f"{portfolioMetrics['expectedAnnualReturn'] * 100:.2f}%"
    )

    column2.metric(
        "Annual Volatility",
        f"{portfolioMetrics['annualVolatility'] * 100:.2f}%"
    )

    column3.metric(
        "Sharpe Ratio",
        f"{portfolioMetrics['sharpeRatio']:.2f}"
    )

    st.subheader("Historical Stock Prices")
    st.line_chart(prices)

    st.subheader("Recommended Portfolio Weights")
    st.bar_chart(weights)

    allocation = pd.DataFrame({
        "Ticker": weights.index,
        "Weight": weights.values,
        "Investment Amount": weights.values * investmentAmount
    })

    allocation["Weight"] = allocation["Weight"].apply(lambda value: f"{value * 100:.2f}%")
    allocation["Investment Amount"] = allocation["Investment Amount"].apply(
        lambda value: f"${value:,.2f}"
    )

    st.dataframe(allocation, width="stretch")

    st.subheader("Model Prediction Results")

    predictionTable = pd.DataFrame({
        "Ticker": list(expectedReturns.keys()),
        "Predicted Next-Day Return": list(expectedReturns.values()),
        "MAE": [modelMetrics[ticker]["mae"] for ticker in expectedReturns],
        "R2": [modelMetrics[ticker]["r2"] for ticker in expectedReturns]
    })

    st.dataframe(predictionTable, width="stretch")

    st.subheader("Strategy Performance Backtest")

    comparison = compareStrategyToBenchmark(
        weights=weights,
        dailyReturns=dailyReturns,
        benchmarkTicker=benchmarkTicker,
        initialValue=investmentAmount
    )

    st.line_chart(comparison)

    if not comparison.empty:
        finalAiValue = comparison["AI Portfolio"].iloc[-1]
        finalBenchmarkValue = comparison["Benchmark"].iloc[-1]

        column4, column5 = st.columns(2)

        column4.metric(
            "Final AI Portfolio Value",
            f"${finalAiValue:,.2f}"
        )

        column5.metric(
            "Final Benchmark Value",
            f"${finalBenchmarkValue:,.2f}"
        )

except Exception as error:
    st.error(f"Something went wrong: {error}")