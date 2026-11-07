import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = os.path.join("data", "training_data.csv")
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
    return df


def build_disease_profiles(df, symptom_cols, min_frequency=0.5):
    profiles = {}
    for disease, group in df.groupby("prognosis"):
        freq = group[symptom_cols].mean()
        common_symptoms = freq[freq >= min_frequency].index.tolist()
        profiles[disease] = common_symptoms
    return profiles


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("Loading dataset...")

    df = load_data()
    symptom_cols = [c for c in df.columns if c != "prognosis"]

    print(
        f"Rows: {len(df)} | Symptoms: {len(symptom_cols)} | "
        f"Diseases: {df['prognosis'].nunique()}"
    )

    X = df[symptom_cols]
    y = df["prognosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Model accuracy: {accuracy:.4f}")

    disease_profiles = build_disease_profiles(df, symptom_cols, min_frequency=0.5)

    print("Saving artifacts...")
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "model.joblib"))

    with open(os.path.join(ARTIFACTS_DIR, "symptom_list.json"), "w") as f:
        json.dump(symptom_cols, f, indent=2)

    with open(os.path.join(ARTIFACTS_DIR, "disease_profiles.json"), "w") as f:
        json.dump(disease_profiles, f, indent=2)

    # CRITICAL: use model.classes_ so order matches predict_proba output
    with open(os.path.join(ARTIFACTS_DIR, "label_classes.json"), "w") as f:
        json.dump(model.classes_.tolist(), f, indent=2)

    print("Done. Artifacts saved to:", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()