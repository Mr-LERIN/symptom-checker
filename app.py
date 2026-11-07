import json
import os
import time

import joblib
import numpy as np
import streamlit as st

from symptom_utils import match_symptoms, symptoms_to_vector
from theme import (inject_theme, nav, hero, section, matched_panel,
                   notice, results, footer)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")

st.set_page_config(page_title="Symptom Checker", page_icon="🫀", layout="wide")
inject_theme()


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


def overlap_score(user_symptoms: set, profile_symptoms: set) -> float:
    if not profile_symptoms:
        return 0.0
    return len(user_symptoms & profile_symptoms) / len(profile_symptoms)


model, symptom_list, disease_profiles, label_classes = load_artifacts()

# ---------- header (text left, clinical tiles right) ----------
nav()
hero()

# ---------- input: two columns, left aligned ----------
section("01", "Describe your symptoms", "Type them, tap them, or both.")

col_left, col_right = st.columns([1.15, 1], gap="large")

with col_left:
    user_input = st.text_area(
        "Symptoms (comma-separated)",
        placeholder="e.g. headache, nausea, fever, chills",
        height=120,
    )
    go = st.button("Analyze symptoms", type="primary")

with col_right:
    picked = st.multiselect(
        "Or pick from the list",
        options=sorted(symptom_list),
        format_func=lambda s: s.replace("_", " "),
        placeholder="Search symptoms...",
    )

# ---------- results ----------
if go:
    if not (user_input.strip() or picked):
        notice("Add at least one symptom: type it above or pick it from the list.")
    else:
        with st.spinner("Analyzing your symptoms..."):
            time.sleep(0.9)  # short beat so the loading state feels intentional
            from_text = list(match_symptoms(user_input, symptom_list)) if user_input.strip() else []
            matched = list(dict.fromkeys(from_text + picked))

        if not matched:
            notice("No symptoms matched. Try rephrasing, or pick from the list.")
        else:
            section("02", "Results", "Ranked by model confidence.")
            matched_panel(matched)

            X_input = symptoms_to_vector(matched, symptom_list)
            probs = model.predict_proba(X_input)[0]
            top5_idx = np.argsort(probs)[::-1][:5]
            user_set = set(matched)

            items = []
            for idx in top5_idx:
                disease = label_classes[idx]
                profile = set(disease_profiles.get(disease, []))
                items.append({
                    "disease": disease,
                    "confidence": float(probs[idx]),
                    "overlap": overlap_score(user_set, profile),
                    "profile": list(profile),
                })
            results(items, matched)

footer()
