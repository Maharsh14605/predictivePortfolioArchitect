import numpy as np
import pandas as pd


def calculateStockScores(expectedReturns, volatility, momentum, riskTolerance):
    scores = {}

    for ticker in expectedReturns:
        expectedReturn = expectedReturns[ticker]
        stockRisk = volatility[ticker]
        stockMomentum = momentum[ticker]

        if stockRisk <= 0 or np.isnan(stockRisk):
            scores[ticker] = 0
        else:
            returnScore = max(expectedReturn, 0)
            momentumScore = max(stockMomentum, 0)

            riskPenalty = stockRisk ** (1.2 - riskTolerance)

            scores[ticker] = (returnScore + 0.5 * momentumScore) / riskPenalty

    return scores


def applyWeightCap(weights, maxWeight=0.4):
    weights = weights.copy()

    for _ in range(10):
        overweightStocks = weights[weights > maxWeight]

        if overweightStocks.empty:
            break

        excessWeight = (overweightStocks - maxWeight).sum()
        weights[overweightStocks.index] = maxWeight

        underweightStocks = weights[weights < maxWeight]

        if underweightStocks.empty:
            break

        weights[underweightStocks.index] += (
            weights[underweightStocks.index] / weights[underweightStocks.index].sum()
        ) * excessWeight

    return weights / weights.sum()


def calculateWeights(expectedReturns, dailyReturns, riskTolerance, maxWeight=0.4):
    volatility = dailyReturns.std() * np.sqrt(252)
    momentum = dailyReturns.tail(21).mean() * 21

    scores = calculateStockScores(
        expectedReturns=expectedReturns,
        volatility=volatility,
        momentum=momentum,
        riskTolerance=riskTolerance
    )

    scoreSeries = pd.Series(scores)

    if riskTolerance < 0.4 and len(scoreSeries) > 3:
        mostVolatileStocks = volatility.sort_values(ascending=False).head(2).index
        scoreSeries = scoreSeries.drop(index=mostVolatileStocks, errors="ignore")

    if scoreSeries.empty or scoreSeries.sum() <= 0:
        lowerRiskStocks = volatility.sort_values().head(min(4, len(volatility))).index
        weights = pd.Series(1 / len(lowerRiskStocks), index=lowerRiskStocks)
    else:
        adjustedScores = scoreSeries ** (0.5 + riskTolerance)
        weights = adjustedScores / adjustedScores.sum()

    weights = applyWeightCap(weights, maxWeight=maxWeight)

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

    portfolioDailyReturns = selectedReturns.dot(weights)
    cumulativeReturns = (1 + portfolioDailyReturns).cumprod()
    runningMax = cumulativeReturns.cummax()
    drawdown = cumulativeReturns / runningMax - 1
    maxDrawdown = float(drawdown.min())

    valueAtRisk95 = float(portfolioDailyReturns.quantile(0.05))

    return {
        "expectedAnnualReturn": portfolioReturn,
        "annualVolatility": portfolioVolatility,
        "sharpeRatio": sharpeRatio,
        "maxDrawdown": maxDrawdown,
        "valueAtRisk95": valueAtRisk95
    }


def calculateCorrelationMatrix(dailyReturns):
    return dailyReturns.corr()