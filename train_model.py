# /// Symptom_checkerCLI
' This is a command-line interface (CLI) for the Symptom Checker application. It allows users to input their symptoms and receive possible diagnoses based on a trained machine learning model.,'

import json
import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = os.path.join("data", "training_data.csv")
ARTIFACTS_DIR = os.path.join("artifacts")


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Drop stray empty/unnamed columns from the source CSV
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
    return df
if __name__ == "__main__":
    df = load_data()
    print("Data loaded successfully!")
    print("Number of rows:", len(df))
    print("Number of columns:", len(df.columns))
    print("First 5 column names:", list(df.columns[:5]))
    print("Last column (should be prognosis):", df.columns[-1])