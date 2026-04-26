import sqlite3
from pathlib import Path

import pandas as pd


databasePath = Path("data/portfolioData.db")


def getConnection():
    databasePath.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(databasePath)


def initializeDatabase():
    connection = getConnection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stockPrices (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL NOT NULL,
            PRIMARY KEY (ticker, date)
        )
    """)

    connection.commit()
    connection.close()


def savePricesToDatabase(prices):
    initializeDatabase()

    connection = getConnection()
    cursor = connection.cursor()

    records = []

    for date, row in prices.iterrows():
        for ticker, closePrice in row.items():
            if pd.notna(closePrice):
                records.append((ticker, str(date.date()), float(closePrice)))

    cursor.executemany("""
        INSERT OR REPLACE INTO stockPrices (ticker, date, close)
        VALUES (?, ?, ?)
    """, records)

    connection.commit()
    connection.close()


def loadPricesFromDatabase(tickers):
    initializeDatabase()

    connection = getConnection()

    placeholders = ",".join(["?"] * len(tickers))

    query = f"""
        SELECT ticker, date, close
        FROM stockPrices
        WHERE ticker IN ({placeholders})
        ORDER BY date
    """

    stockData = pd.read_sql_query(query, connection, params=tickers)

    connection.close()

    if stockData.empty:
        return pd.DataFrame()

    stockData["date"] = pd.to_datetime(stockData["date"])

    prices = stockData.pivot(index="date", columns="ticker", values="close")
    prices = prices.sort_index()
    prices = prices.ffill()

    return prices