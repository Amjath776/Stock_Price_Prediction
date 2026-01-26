def preprocess_data(df):
    df = df.copy()

    df["Return"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()

    df.dropna(inplace=True)

    X = df[["Open", "High", "Low", "Volume", "Return", "MA_5", "MA_10"]]
    y = df["Close"].values.ravel()


    return X, y
