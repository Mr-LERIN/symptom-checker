import json
import os

import joblib
import numpy as np

from symptom_utils import match_symptoms, symptoms_to_vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")


def load_artifacts():
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "model.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "symptom_list.json")) as f:
        symptom_list = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "disease_profiles.json")) as f:
        disease_profiles = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "label_classes.json")) as f:
        label_classes = json.load(f)
    return model, symptom_list, disease_profiles, label_classes


def jaccard_overlap(user_symptoms: set, profile_symptoms: set) -> float:
    if not profile_symptoms:
        return 0.0
    return len(user_symptoms & profile_symptoms) / len(profile_symptoms)


def explain_prediction(model, symptom_list, disease_profiles, label_classes, matched_symptoms):
    user_set = set(matched_symptoms)
    X_input = symptoms_to_vector(matched_symptoms, symptom_list)
    probs = model.predict_proba(X_input)[0]

    top5_idx = np.argsort(probs)[::-1][:5]

    print("\nTop possible conditions:")
    print("-" * 50)
    for rank, idx in enumerate(top5_idx, 1):
        disease = label_classes[idx]
        confidence = probs[idx]
        profile = set(disease_profiles.get(disease, []))
        overlap = jaccard_overlap(user_set, profile)

        print(f"{rank}. {disease}")
        print(f"   Model confidence: {confidence:.1%}")
        print(f"   Symptom overlap with typical profile: {overlap:.1%}")
        print(f"   Typical symptoms: {', '.join(sorted(profile)) or 'N/A'}")
        print()


def main():
    if not os.path.exists(os.path.join(ARTIFACTS_DIR, "model.joblib")):
        print("No trained model found. Run Symptom_checkerCLI.py first.")
        return

    print("Loading model and artifacts...")
    model, symptom_list, disease_profiles, label_classes = load_artifacts()
    print(f"Loaded. Knows {len(symptom_list)} symptoms and {len(label_classes)} diseases.\n")

    print("Type your symptoms (comma-separated).")
    print("Commands: 'list' = show all symptoms, 'quit' = exit.\n")

    while True:
        user_input = input("Symptoms: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "list":
            print(f"\nRecognized symptoms ({len(symptom_list)}):")
            for s in sorted(symptom_list):
                print(f"  - {s.replace('_', ' ')}")
            print()
            continue
        if not user_input:
            continue

        matched = match_symptoms(user_input, symptom_list)
        print(f"\nMatched {len(matched)} symptom(s): {', '.join(matched) or 'None'}")

        if not matched:
            print("No symptoms could be matched. Try different wording.\n")
            continue

        explain_prediction(model, symptom_list, disease_profiles, label_classes, matched)


if __name__ == "__main__":
    main()