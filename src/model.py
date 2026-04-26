from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


featureColumns = [
    "dailyReturn",
    "movingAverage7",
    "movingAverage21",
    "returnAverage7",
    "volatility21",
    "priceVsMovingAverage7",
    "priceVsMovingAverage21"
]


modelDirectory = Path("models/savedModels")


def trainModelForStock(stockData):
    splitIndex = int(len(stockData) * 0.8)

    trainingData = stockData.iloc[:splitIndex]
    testingData = stockData.iloc[splitIndex:]

    xTrain = trainingData[featureColumns]
    yTrain = trainingData["target"]

    xTest = testingData[featureColumns]
    yTest = testingData["target"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(xTrain, yTrain)

    predictions = model.predict(xTest)

    metrics = {
        "mae": mean_absolute_error(yTest, predictions),
        "r2": r2_score(yTest, predictions)
    }

    latestFeatures = stockData[featureColumns].iloc[[-1]]
    expectedReturn = model.predict(latestFeatures)[0]

    return model, expectedReturn, metrics


def saveModel(model, ticker):
    modelDirectory.mkdir(parents=True, exist_ok=True)

    filePath = modelDirectory / f"{ticker.replace('.', '_')}_model.joblib"

    joblib.dump(model, filePath)


def loadModel(ticker):
    filePath = modelDirectory / f"{ticker.replace('.', '_')}_model.joblib"

    if not filePath.exists():
        return None

    return joblib.load(filePath)


def predictWithSavedModel(ticker, stockData):
    model = loadModel(ticker)

    if model is None:
        model, expectedReturn, metrics = trainModelForStock(stockData)
        saveModel(model, ticker)

        return expectedReturn, metrics

    latestFeatures = stockData[featureColumns].iloc[[-1]]
    expectedReturn = model.predict(latestFeatures)[0]

    return expectedReturn, None