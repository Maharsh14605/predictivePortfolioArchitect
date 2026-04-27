from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


featureColumns = [
    "dailyReturn",
    "return5",
    "return10",
    "return21",
    "movingAverage7",
    "movingAverage21",
    "movingAverage50",
    "movingAverage200",
    "priceVsMovingAverage7",
    "priceVsMovingAverage21",
    "priceVsMovingAverage50",
    "priceVsMovingAverage200",
    "volatility7",
    "volatility21",
    "volatility63",
    "returnAverage7",
    "returnAverage21",
    "rsi14",
    "macd",
    "macdSignal",
    "macdHistogram",
    "drawdown"
]


modelDirectory = Path("models/savedModels")


def createCandidateModels():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=3,
            random_state=42
        ),
        "Ridge Regression": Ridge(alpha=1.0)
    }


def trainModelForStock(stockData):
    splitIndex = int(len(stockData) * 0.8)

    trainingData = stockData.iloc[:splitIndex]
    testingData = stockData.iloc[splitIndex:]

    xTrain = trainingData[featureColumns]
    yTrain = trainingData["target"]

    xTest = testingData[featureColumns]
    yTest = testingData["target"]

    candidateModels = createCandidateModels()

    bestModel = None
    bestModelName = None
    bestMetrics = None
    bestMae = float("inf")

    modelResults = []

    baselinePrediction = [yTrain.mean()] * len(yTest)

    baselineMetrics = {
        "modelName": "Baseline Average",
        "mae": mean_absolute_error(yTest, baselinePrediction),
        "rmse": mean_squared_error(yTest, baselinePrediction) ** 0.5,
        "r2": r2_score(yTest, baselinePrediction)
    }

    modelResults.append(baselineMetrics)

    for modelName, model in candidateModels.items():
        model.fit(xTrain, yTrain)

        predictions = model.predict(xTest)

        metrics = {
            "modelName": modelName,
            "mae": mean_absolute_error(yTest, predictions),
            "rmse": mean_squared_error(yTest, predictions) ** 0.5,
            "r2": r2_score(yTest, predictions)
        }

        modelResults.append(metrics)

        if metrics["mae"] < bestMae:
            bestMae = metrics["mae"]
            bestModel = model
            bestModelName = modelName
            bestMetrics = metrics

    latestFeatures = stockData[featureColumns].iloc[[-1]]
    expectedReturn = bestModel.predict(latestFeatures)[0]

    modelResultsTable = pd.DataFrame(modelResults)

    return bestModel, expectedReturn, bestMetrics, bestModelName, modelResultsTable


def saveModel(model, ticker):
    modelDirectory.mkdir(parents=True, exist_ok=True)

    filePath = modelDirectory / f"{ticker.replace('.', '_')}_model.joblib"

    joblib.dump(model, filePath)


def loadModel(ticker):
    filePath = modelDirectory / f"{ticker.replace('.', '_')}_model.joblib"

    if not filePath.exists():
        return None

    return joblib.load(filePath)


def getFeatureImportance(model):
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    importanceTable = pd.DataFrame({
        "Feature": featureColumns,
        "Importance": model.feature_importances_
    })

    importanceTable = importanceTable.sort_values(
        by="Importance",
        ascending=False
    )

    return importanceTable