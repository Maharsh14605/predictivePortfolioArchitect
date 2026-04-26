import numpy as np
import pandas as pd


def calculateStockScores(expectedReturns, volatility):
    scores = {}

    for ticker in expectedReturns:
        expectedReturn = expectedReturns[ticker]
        stockRisk = volatility[ticker]

        if stockRisk <= 0 or np.isnan(stockRisk):
            scores[ticker] = 0
        else:
            scores[ticker] = max(expectedReturn, 0) / stockRisk

    return scores


def calculateWeights(expectedReturns, dailyReturns, riskTolerance):
    volatility = dailyReturns.std() * np.sqrt(252)

    scores = calculateStockScores(expectedReturns, volatility)
    scoreSeries = pd.Series(scores)

    if riskTolerance < 0.4 and len(scoreSeries) > 2:
        mostVolatileStocks = volatility.sort_values(ascending=False).head(2).index
        scoreSeries = scoreSeries.drop(index=mostVolatileStocks, errors="ignore")

    if scoreSeries.empty:
        scoreSeries = pd.Series(1, index=dailyReturns.columns)

    if scoreSeries.sum() <= 0:
        weights = pd.Series(1 / len(scoreSeries), index=scoreSeries.index)
    else:
        adjustedScores = scoreSeries ** riskTolerance
        weights = adjustedScores / adjustedScores.sum()

    return weights.sort_values(ascending=False)


def calculatePortfolioMetrics(weights, dailyReturns, riskFreeRate=0.03):
    selectedReturns = dailyReturns[weights.index]

    meanDailyReturns = selectedReturns.mean()
    annualReturns = meanDailyReturns * 252

    portfolioReturn = float((weights * annualReturns).sum())

    covarianceMatrix = selectedReturns.cov() * 252
    portfolioVariance = np.dot(weights.T, np.dot(covarianceMatrix, weights))
    portfolioVolatility = float(np.sqrt(portfolioVariance))

    if portfolioVolatility == 0:
        sharpeRatio = 0
    else:
        sharpeRatio = (portfolioReturn - riskFreeRate) / portfolioVolatility

    return {
        "expectedAnnualReturn": portfolioReturn,
        "annualVolatility": portfolioVolatility,
        "sharpeRatio": sharpeRatio
    }