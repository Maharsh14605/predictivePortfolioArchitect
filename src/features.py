import numpy as np
import pandas as pd


def calculateRsi(priceSeries, window=14):
    priceDifference = priceSeries.diff()

    gain = priceDifference.clip(lower=0)
    loss = -priceDifference.clip(upper=0)

    averageGain = gain.rolling(window=window).mean()
    averageLoss = loss.rolling(window=window).mean()

    relativeStrength = averageGain / averageLoss
    rsi = 100 - (100 / (1 + relativeStrength))

    return rsi


def calculateMacd(priceSeries):
    ema12 = priceSeries.ewm(span=12, adjust=False).mean()
    ema26 = priceSeries.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    macdSignal = macd.ewm(span=9, adjust=False).mean()
    macdHistogram = macd - macdSignal

    return macd, macdSignal, macdHistogram


def calculateDrawdown(priceSeries):
    rollingMax = priceSeries.cummax()
    drawdown = priceSeries / rollingMax - 1

    return drawdown


def createFeaturesForStock(priceSeries):
    stockFeatures = pd.DataFrame()

    stockFeatures["close"] = priceSeries
    stockFeatures["dailyReturn"] = stockFeatures["close"].pct_change()

    stockFeatures["return5"] = stockFeatures["close"].pct_change(5)
    stockFeatures["return10"] = stockFeatures["close"].pct_change(10)
    stockFeatures["return21"] = stockFeatures["close"].pct_change(21)

    stockFeatures["movingAverage7"] = stockFeatures["close"].rolling(window=7).mean()
    stockFeatures["movingAverage21"] = stockFeatures["close"].rolling(window=21).mean()
    stockFeatures["movingAverage50"] = stockFeatures["close"].rolling(window=50).mean()
    stockFeatures["movingAverage200"] = stockFeatures["close"].rolling(window=200).mean()

    stockFeatures["priceVsMovingAverage7"] = (
        stockFeatures["close"] / stockFeatures["movingAverage7"] - 1
    )

    stockFeatures["priceVsMovingAverage21"] = (
        stockFeatures["close"] / stockFeatures["movingAverage21"] - 1
    )

    stockFeatures["priceVsMovingAverage50"] = (
        stockFeatures["close"] / stockFeatures["movingAverage50"] - 1
    )

    stockFeatures["priceVsMovingAverage200"] = (
        stockFeatures["close"] / stockFeatures["movingAverage200"] - 1
    )

    stockFeatures["volatility7"] = stockFeatures["dailyReturn"].rolling(window=7).std()
    stockFeatures["volatility21"] = stockFeatures["dailyReturn"].rolling(window=21).std()
    stockFeatures["volatility63"] = stockFeatures["dailyReturn"].rolling(window=63).std()

    stockFeatures["returnAverage7"] = stockFeatures["dailyReturn"].rolling(window=7).mean()
    stockFeatures["returnAverage21"] = stockFeatures["dailyReturn"].rolling(window=21).mean()

    stockFeatures["rsi14"] = calculateRsi(stockFeatures["close"], window=14)

    macd, macdSignal, macdHistogram = calculateMacd(stockFeatures["close"])
    stockFeatures["macd"] = macd
    stockFeatures["macdSignal"] = macdSignal
    stockFeatures["macdHistogram"] = macdHistogram

    stockFeatures["drawdown"] = calculateDrawdown(stockFeatures["close"])

    stockFeatures["target"] = stockFeatures["close"].shift(-21) / stockFeatures["close"] - 1

    stockFeatures = stockFeatures.replace([np.inf, -np.inf], np.nan)
    stockFeatures = stockFeatures.dropna()

    return stockFeatures


def createFeatureData(prices):
    featureData = {}

    for ticker in prices.columns:
        featureData[ticker] = createFeaturesForStock(prices[ticker])

    return featureData