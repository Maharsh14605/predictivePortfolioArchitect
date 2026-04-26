import pandas as pd


def createFeaturesForStock(priceSeries):
    stockFeatures = pd.DataFrame()

    stockFeatures["close"] = priceSeries
    stockFeatures["dailyReturn"] = stockFeatures["close"].pct_change()

    stockFeatures["movingAverage7"] = stockFeatures["close"].rolling(window=7).mean()
    stockFeatures["movingAverage21"] = stockFeatures["close"].rolling(window=21).mean()

    stockFeatures["returnAverage7"] = stockFeatures["dailyReturn"].rolling(window=7).mean()
    stockFeatures["volatility21"] = stockFeatures["dailyReturn"].rolling(window=21).std()

    stockFeatures["priceVsMovingAverage7"] = (
        stockFeatures["close"] / stockFeatures["movingAverage7"] - 1
    )

    stockFeatures["priceVsMovingAverage21"] = (
        stockFeatures["close"] / stockFeatures["movingAverage21"] - 1
    )

    stockFeatures["target"] = stockFeatures["dailyReturn"].shift(-1)

    stockFeatures = stockFeatures.dropna()

    return stockFeatures


def createFeatureData(prices):
    featureData = {}

    for ticker in prices.columns:
        featureData[ticker] = createFeaturesForStock(prices[ticker])

    return featureData