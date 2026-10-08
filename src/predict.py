import argparse
import joblib

from src.model_utils import predict_single_text


def main():
    parser = argparse.ArgumentParser(description="Predict whether a news article is real or fake.")
    parser.add_argument("--model", type=str, default="models/fake_news_model.joblib", help="Path to the trained model.")
    parser.add_argument("--text", type=str, default="", help="News article text to classify.")
    args = parser.parse_args()

    model = joblib.load(args.model)

    if args.text:
        article = args.text
    else:
        article = input("Enter news article text: ")

    prediction = predict_single_text(model, article)
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
