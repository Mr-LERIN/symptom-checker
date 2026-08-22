# /// Symptom_checkerCLI
# This is a command-line interface (CLI) for the Symptom Checker application.
# It allows users to input their symptoms and receive possible diagnoses
# based on a trained machine learning model.

import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths (you are already inside symptom_checker_v2)
DATA_PATH = os.path.join("data", "training_data.csv")
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("Paths are set correctly.")
print("Data path:", DATA_PATH)
print("Artifacts folder:", ARTIFACTS_DIR)


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Drop stray empty/unnamed columns from the csv
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
    return df


def build_disease_profiles(df, symptom_cols, min_frequency=0.5):
    """
    For each disease, find the symptoms that show up in at least
    'min_frequency' fraction of that disease's rows. This becomes the
    'typical symptom set' used for the explainable overlap score.
    """
    profiles = {}
    for disease, group in df.groupby("prognosis"):
        freq = group[symptom_cols].mean()  # fraction of rows where symptom=1
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
    #Train the model
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    #Evaluate the model
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Model accuracy: {accuracy:.4f}")

    #Build disease typical symptom profiles for explainable overlap score
    print
    disease_profiles = build_disease_profiles(df, symptom_cols, min_frequency=0.5)

    #Save the model and disease profiles
    print("Saving Artifacts...")
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "model.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "symptom_list.json"), "w") as f:
        json.dump(symptom_cols, f, indent=2)

    with open(os.path.join(ARTIFACTS_DIR, "disease_profiles.json"), "w") as f:
        json.dump(disease_profiles, f, indent=2)

    with open(os.path.join(ARTIFACTS_DIR, "label_classes.json"), "w") as f:
        json.dump(list(df["prognosis"].unique()), f, indent=2)

        print("Done. Artifacts saved to:", ARTIFACTS_DIR)
if  __name__ == "__main__":
    main()