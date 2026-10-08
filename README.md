# Fake News Detector

This project classifies news articles as real or fake using NLP, TF-IDF, and a Random Forest classifier.

It is intentionally kept separate from the real-estate project.

## Project goals
- Detect fake versus real articles from text content
- Use TF-IDF for text vectorization
- Train a Random Forest model for classification
- Allow prediction on new article text

## Repository structure
- `data/sample_news.csv` — small sample dataset for quick testing
- `src/model_utils.py` — preprocessing, training, and evaluation helpers
- `src/train_model.py` — train a model from a CSV file
- `src/predict.py` — run predictions on new text
- `models/` — output folder for the trained model

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

```bash
python src/train_model.py --data data/sample_news.csv --output models/fake_news_model.joblib
```

## Predict new article text

```bash
python src/predict.py --model models/fake_news_model.joblib --text "Stock prices surged after the central bank announced a surprise rate cut. Analysts described the move as positive for markets."
```

## Notes
- The sample dataset is intentionally small and is meant for demonstration.
- Replace it with a real labeled dataset for production use.
- The model pipeline saves both vectorizer and classifier together for easy reuse.
