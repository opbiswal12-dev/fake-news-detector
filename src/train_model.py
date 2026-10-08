import argparse
import os

from src.model_utils import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description="Train a fake news detector model.")
    parser.add_argument("--data", type=str, default="data/sample_news.csv", help="Path to the CSV dataset.")
    parser.add_argument(
        "--output",
        type=str,
        default="models/fake_news_model.joblib",
        help="Where to save the trained model.",
    )
    args = parser.parse_args()

    _, metrics = train_and_evaluate(args.data, args.output)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Classification report:")
    print(metrics["classification_report"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
