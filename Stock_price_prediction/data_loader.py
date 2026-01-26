import yfinance as yf

def load_stock_data(ticker="AAPL", start="2018-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end)
    df = df.reset_index()
    return df
