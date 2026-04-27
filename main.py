from src.dataLoader import calculateDailyReturns
from src.database import loadPricesFromDatabase
from src.features import createFeatureData
from src.model import trainModelForStock
from src.portfolio import calculateWeights, calculatePortfolioMetrics


def main():
    tickers = ["RY.TO", "TD.TO", "AAPL", "MSFT"]

    prices = loadPricesFromDatabase(tickers)

    if prices.empty:
        print("No database data found. Run: python updateDaily.py")
        return

    dailyReturns = calculateDailyReturns(prices)
    featureData = createFeatureData(prices)

    expectedReturns = {}

    for ticker, stockData in featureData.items():
        model, expectedReturn, metrics, modelName, modelResultsTable = trainModelForStock(stockData)

        expectedReturns[ticker] = expectedReturn

        print(f"{ticker}")
        print(f"  Best model: {modelName}")
        print(f"  Predicted 21-day return: {expectedReturn:.5f}")
        print(f"  MAE: {metrics['mae']:.5f}")
        print(f"  RMSE: {metrics['rmse']:.5f}")
        print(f"  R2: {metrics['r2']:.5f}")

    weights = calculateWeights(
        expectedReturns=expectedReturns,
        dailyReturns=dailyReturns,
        riskTolerance=0.5
    )

    metrics = calculatePortfolioMetrics(weights, dailyReturns)

    print("\nRecommended weights:")
    print(weights)

    print("\nPortfolio metrics:")
    print(metrics)


if __name__ == "__main__":
    main()