import pandas as pd

from src.dataLoader import downloadPrices, calculateDailyReturns


def backtestPortfolio(weights, dailyReturns, initialValue=10000):
    selectedReturns = dailyReturns[weights.index]

    portfolioDailyReturns = selectedReturns.dot(weights)
    portfolioValues = (1 + portfolioDailyReturns).cumprod() * initialValue

    return portfolioValues


def backtestBenchmark(benchmarkTicker="^GSPC", period="1y", initialValue=10000):
    benchmarkPrices = downloadPrices([benchmarkTicker], period=period)
    benchmarkReturns = calculateDailyReturns(benchmarkPrices)

    benchmarkValues = (1 + benchmarkReturns.iloc[:, 0]).cumprod() * initialValue

    return benchmarkValues


def compareStrategyToBenchmark(weights, dailyReturns, benchmarkTicker="^GSPC", initialValue=10000):
    oneYearReturns = dailyReturns.tail(252)

    portfolioValues = backtestPortfolio(
        weights=weights,
        dailyReturns=oneYearReturns,
        initialValue=initialValue
    )

    benchmarkValues = backtestBenchmark(
        benchmarkTicker=benchmarkTicker,
        period="1y",
        initialValue=initialValue
    )

    comparison = pd.DataFrame({
        "AI Portfolio": portfolioValues,
        "Benchmark": benchmarkValues
    })

    comparison = comparison.dropna()

    return comparison