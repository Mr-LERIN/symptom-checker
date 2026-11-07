import json
import os

import joblib
import numpy as np
import streamlit as st

from symptom_utils import match_symptoms, symptoms_to_vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")


@st.cache_resource
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


st.set_page_config(page_title="Symptom Checker", page_icon="🩺")
st.title("🩺 Symptom Checker")
st.caption("Educational project — not a substitute for professional medical advice.")

model, symptom_list, disease_profiles, label_classes = load_artifacts()

with st.expander("See all recognized symptoms"):
    st.write(", ".join(s.replace("_", " ") for s in sorted(symptom_list)))

user_input = st.text_area(
    "Describe your symptoms (comma-separated):",
    placeholder="e.g. headache, nausea, fever, chills",
)

if st.button("Check", type="primary") and user_input:
    matched = match_symptoms(user_input, symptom_list)

    if matched:
        st.success(f"Matched symptoms: {', '.join(matched)}")
    else:
        st.warning("No symptoms matched. Try rephrasing.")
        st.stop()

    X_input = symptoms_to_vector(matched, symptom_list)
    probs = model.predict_proba(X_input)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    user_set = set(matched)

    st.subheader("Top possible conditions")

    for rank, idx in enumerate(top5_idx, 1):
        disease = label_classes[idx]
        confidence = probs[idx]
        profile = set(disease_profiles.get(disease, []))
        overlap = jaccard_overlap(user_set, profile)

        with st.expander(f"{rank}. {disease} — {confidence:.1%} confidence"):
            st.write(f"**Typical symptoms:** {', '.join(sorted(profile)) or 'N/A'}")
            st.write(f"**Your symptom overlap:** {overlap:.1%}")
            st.progress(min(overlap, 1.0), text=f"Overlap: {overlap:.0%}")

st.divider()
st.caption("Built with Streamlit · Synthetic dataset · Not for clinical use")