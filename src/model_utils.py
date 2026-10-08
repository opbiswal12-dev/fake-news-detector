import re
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


LABEL_MAP = {
    "real": 1,
    "fake": 0,
    "true": 1,
    "false": 0,
    "1": 1,
    "0": 0,
}


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(csv_path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in dataset.")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in dataset.")
    return df[[text_col, label_col]].copy()


def prepare_labels(series: pd.Series) -> pd.Series:
    return series.map(lambda value: LABEL_MAP.get(str(value).strip().lower(), int(value)))


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=5000,
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_evaluate(csv_path: str, output_path: str) -> Tuple[Pipeline, dict]:
    df = load_dataset(csv_path)
    df["text"] = df["text"].apply(clean_text)
    df["label"] = prepare_labels(df["label"])

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["fake", "real"], output_dict=True)
    matrix = confusion_matrix(y_test, predictions)

    model_dir = "/".join(output_path.split("/")[:-1])
    if model_dir:
        import os
        os.makedirs(model_dir, exist_ok=True)

    joblib.dump(pipeline, output_path)

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
    return pipeline, metrics


def predict_single_text(model: Pipeline, text: str) -> str:
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    return "real" if prediction == 1 else "fake"
