from src.dataLoader import downloadPrices, calculateDailyReturns
from src.database import savePricesToDatabase, loadPricesFromDatabase
from src.features import createFeatureData
from src.model import trainModelForStock, saveModel


defaultTickers = [
    "RY.TO",
    "TD.TO",
    "BNS.TO",
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL"
]


def updateMarketData(tickers):
    print("Downloading latest stock data...")

    prices = downloadPrices(tickers, period="5y")

    print("Saving prices to database...")

    savePricesToDatabase(prices)

    return prices


def retrainModels(tickers):
    print("Loading prices from database...")

    prices = loadPricesFromDatabase(tickers)

    if prices.empty:
        raise ValueError("No prices found in database.")

    dailyReturns = calculateDailyReturns(prices)
    featureData = createFeatureData(prices)

    print("Retraining models...")

    results = {}

    for ticker, stockData in featureData.items():
        model, expectedReturn, metrics = trainModelForStock(stockData)

        saveModel(model, ticker)

        results[ticker] = {
            "expectedReturn": expectedReturn,
            "mae": metrics["mae"],
            "r2": metrics["r2"]
        }

        print(
            f"{ticker}: expectedReturn={expectedReturn:.5f}, "
            f"mae={metrics['mae']:.5f}, r2={metrics['r2']:.5f}"
        )

    return results


def main():
    updateMarketData(defaultTickers)
    retrainModels(defaultTickers)

    print("Daily update completed successfully.")


if __name__ == "__main__":
    main()