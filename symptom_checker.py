import json
import os
import joblib
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

def main():
    print("Loading model and artifacts...")
    model, symptom_list, disease_profiles, label_classes = load_artifacts()
    print(f"Model loaded. It knows {len(symptom_list)} symptoms and {len(label_classes)} diseases.\n")

    user_input = input("Describe your symptoms (separated by commas):\n> ")
    matched = match_symptoms(user_input, symptom_list)
    print("\nMatched symptoms:", matched)

    if not matched:
        print("No symptoms could be matched. Try different wording.")
        return

    vector = symptoms_to_vector(matched, symptom_list)
    print("Feature vector created (first 10 values):", vector[:10])

    prediction = model.predict([vector])[0]
    probs = model.predict_proba([vector])[0]
    confidence = max(probs)

    print(f"\nPredicted condition: {prediction}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main()