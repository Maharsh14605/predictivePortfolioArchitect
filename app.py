import streamlit as st
import pandas as pd

from src.backtest import compareStrategyToBenchmark
from src.dataLoader import downloadPrices, calculateDailyReturns
from src.database import loadPricesFromDatabase, savePricesToDatabase
from src.features import createFeatureData
from src.model import trainModelForStock, saveModel, getFeatureImportance
from src.portfolio import (
    calculateWeights,
    calculatePortfolioMetrics,
    calculateCorrelationMatrix
)


st.set_page_config(
    page_title="Predictive Portfolio Architect",
    page_icon="📈",
    layout="wide"
)


st.markdown("""
<style>
.main-header {
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.sub-header {
    color: #6c757d;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1.1rem;
    border-radius: 0.9rem;
    border: 1px solid #e9ecef;
}
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 0.8rem;
}
.warning-box {
    background-color: #fff3cd;
    padding: 0.9rem;
    border-radius: 0.7rem;
    border: 1px solid #ffeeba;
    color: #664d03;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-header">Predictive Portfolio Architect</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Machine learning portfolio allocation with daily market data, risk analysis, and benchmark backtesting.</div>',
    unsafe_allow_html=True
)


@st.cache_data(ttl=86400)
def loadData(tickers):
    prices = loadPricesFromDatabase(tickers)

    missingData = prices.empty or any(ticker not in prices.columns for ticker in tickers)

    if missingData:
        prices = downloadPrices(tickers, period="5y")
        savePricesToDatabase(prices)

    dailyReturns = calculateDailyReturns(prices)

    return prices, dailyReturns


def formatPercent(value):
    return f"{value * 100:.2f}%"


def formatMoney(value):
    return f"${value:,.2f}"


with st.sidebar:
    st.header("Portfolio Settings")

    tickerInput = st.text_area(
        "Stock Tickers",
        value="RY.TO, TD.TO, BNS.TO, AAPL, MSFT, AMZN, GOOGL, NVDA",
        height=90
    )

    investmentAmount = st.number_input(
        "Investment Amount",
        min_value=1000,
        value=10000,
        step=1000
    )

    riskTolerance = st.slider(
        "Risk Tolerance",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    maxWeight = st.slider(
        "Max Weight Per Stock",
        min_value=0.10,
        max_value=0.60,
        value=0.40,
        step=0.05
    )

    benchmarkTicker = st.selectbox(
        "Benchmark",
        options=["^GSPC", "^GSPTSE"],
        index=0
    )

    runButton = st.button("Build Portfolio", use_container_width=True)


tickers = [
    ticker.strip().upper()
    for ticker in tickerInput.replace("\n", ",").split(",")
    if ticker.strip()
]


if not runButton:
    st.info("Enter tickers in the sidebar and click Build Portfolio.")
    st.stop()


if len(tickers) < 2:
    st.warning("Please enter at least two tickers.")
    st.stop()


try:
    with st.spinner("Loading market data and training models..."):
        prices, dailyReturns = loadData(tickers)
        featureData = createFeatureData(prices)

        expectedReturns = {}
        modelMetrics = {}
        bestModels = {}
        allModelResults = {}
        featureImportanceTables = {}

        for ticker, stockData in featureData.items():
            model, expectedReturn, metrics, modelName, modelResultsTable = trainModelForStock(stockData)

            saveModel(model, ticker)

            expectedReturns[ticker] = expectedReturn
            modelMetrics[ticker] = metrics
            bestModels[ticker] = modelName
            allModelResults[ticker] = modelResultsTable
            featureImportanceTables[ticker] = getFeatureImportance(model)

        weights = calculateWeights(
            expectedReturns=expectedReturns,
            dailyReturns=dailyReturns,
            riskTolerance=riskTolerance,
            maxWeight=maxWeight
        )

        portfolioMetrics = calculatePortfolioMetrics(
            weights=weights,
            dailyReturns=dailyReturns
        )

    overviewTab, allocationTab, riskTab, modelTab, backtestTab = st.tabs([
        "Overview",
        "Allocation",
        "Risk Analysis",
        "Model Insights",
        "Backtest"
    ])

    with overviewTab:
        st.markdown('<div class="section-title">Portfolio Summary</div>', unsafe_allow_html=True)

        column1, column2, column3, column4, column5 = st.columns(5)

        column1.metric("Expected Annual Return", formatPercent(portfolioMetrics["expectedAnnualReturn"]))
        column2.metric("Annual Volatility", formatPercent(portfolioMetrics["annualVolatility"]))
        column3.metric("Sharpe Ratio", f"{portfolioMetrics['sharpeRatio']:.2f}")
        column4.metric("Max Drawdown", formatPercent(portfolioMetrics["maxDrawdown"]))
        column5.metric("Daily VaR 95%", formatPercent(portfolioMetrics["valueAtRisk95"]))

        st.markdown('<div class="section-title">Historical Price Trends</div>', unsafe_allow_html=True)
        st.line_chart(prices)

        st.markdown('<div class="warning-box">This project is for educational purposes only and is not financial advice.</div>', unsafe_allow_html=True)

    with allocationTab:
        st.markdown('<div class="section-title">Recommended Allocation</div>', unsafe_allow_html=True)

        allocation = pd.DataFrame({
            "Ticker": weights.index,
            "Weight": weights.values,
            "Investment Amount": weights.values * investmentAmount,
            "Predicted 21-Day Return": [expectedReturns[ticker] for ticker in weights.index],
            "Best Model": [bestModels[ticker] for ticker in weights.index]
        })

        displayAllocation = allocation.copy()
        displayAllocation["Weight"] = displayAllocation["Weight"].apply(formatPercent)
        displayAllocation["Investment Amount"] = displayAllocation["Investment Amount"].apply(formatMoney)
        displayAllocation["Predicted 21-Day Return"] = displayAllocation["Predicted 21-Day Return"].apply(formatPercent)

        column1, column2 = st.columns([1, 1])

        with column1:
            st.bar_chart(weights)

        with column2:
            st.dataframe(displayAllocation, width="stretch")

        csvData = allocation.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Allocation CSV",
            data=csvData,
            file_name="portfolioAllocation.csv",
            mime="text/csv"
        )

    with riskTab:
        st.markdown('<div class="section-title">Risk Analysis</div>', unsafe_allow_html=True)

        volatilityTable = pd.DataFrame({
            "Ticker": dailyReturns.columns,
            "Annual Volatility": dailyReturns.std().values * (252 ** 0.5),
            "Average Daily Return": dailyReturns.mean().values
        })

        volatilityTable["Annual Volatility"] = volatilityTable["Annual Volatility"].apply(formatPercent)
        volatilityTable["Average Daily Return"] = volatilityTable["Average Daily Return"].apply(formatPercent)

        st.dataframe(volatilityTable, width="stretch")

        st.markdown('<div class="section-title">Correlation Matrix</div>', unsafe_allow_html=True)

        correlationMatrix = calculateCorrelationMatrix(dailyReturns[weights.index])
        st.dataframe(correlationMatrix, width="stretch")

        st.caption("Correlation helps show whether selected assets move together or provide diversification.")

    with modelTab:
        st.markdown('<div class="section-title">Model Performance by Stock</div>', unsafe_allow_html=True)

        predictionTable = pd.DataFrame({
            "Ticker": list(expectedReturns.keys()),
            "Best Model": [bestModels[ticker] for ticker in expectedReturns],
            "Predicted 21-Day Return": list(expectedReturns.values()),
            "MAE": [modelMetrics[ticker]["mae"] for ticker in expectedReturns],
            "RMSE": [modelMetrics[ticker]["rmse"] for ticker in expectedReturns],
            "R2": [modelMetrics[ticker]["r2"] for ticker in expectedReturns]
        })

        displayPredictionTable = predictionTable.copy()
        displayPredictionTable["Predicted 21-Day Return"] = displayPredictionTable[
            "Predicted 21-Day Return"
        ].apply(formatPercent)

        st.dataframe(displayPredictionTable, width="stretch")

        selectedTicker = st.selectbox(
            "Select ticker for model comparison",
            options=list(allModelResults.keys())
        )

        st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
        st.dataframe(allModelResults[selectedTicker], width="stretch")

        featureImportance = featureImportanceTables[selectedTicker]

        if not featureImportance.empty:
            st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
            st.bar_chart(featureImportance.set_index("Feature").head(10))
        else:
            st.info("Feature importance is not available for this model type.")

    with backtestTab:
        st.markdown('<div class="section-title">Strategy Performance Backtest</div>', unsafe_allow_html=True)

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

            column1, column2, column3 = st.columns(3)

            column1.metric("Final AI Portfolio Value", formatMoney(finalAiValue))
            column2.metric("Final Benchmark Value", formatMoney(finalBenchmarkValue))
            column3.metric("Difference", formatMoney(finalAiValue - finalBenchmarkValue))

except Exception as error:
    st.error(f"Something went wrong: {error}")