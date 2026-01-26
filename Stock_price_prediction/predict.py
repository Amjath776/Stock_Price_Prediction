from data_loader import load_stock_data
from preprocessing import preprocess_data
from train_model import train_model
from evaluate import evaluate_model

def main():
    df = load_stock_data("AAPL")
    X, y = preprocess_data(df)

    model, X_test, y_test = train_model(X, y)
    predictions = evaluate_model(model, X_test, y_test)

    print("Predicted price:", predictions[-1])
    print("Actual price:", y_test[-1])


if __name__ == "__main__":
    main()
