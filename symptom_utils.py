import difflib

SYNONYMS = {
    "fever": ["high temperature", "pyrexia", "temperature"],
    "headache": ["head pain", "migraine"],
    "nausea": ["feeling sick", "queasy"],
    "vomiting": ["throwing up", "puking"],
    "fatigue": ["tiredness", "exhaustion", "feeling tired"],
    "cough": ["coughing"],
    "sore throat": ["throat pain", "painful throat"],
    "runny nose": ["rhinorrhea", "nose running"],
    "shortness of breath": ["breathlessness", "difficulty breathing", "dyspnea"],
    "chest pain": ["pain in chest"],
    "abdominal pain": ["stomach pain", "belly pain", "tummy pain"],
    "diarrhea": ["loose stools", "diarrhoea"],
    "constipation": ["difficulty passing stool"],
    "dizziness": ["lightheaded", "giddiness"],
    "rash": ["skin rash", "eruption"],
}

def normalize(text: str) -> str:
    return text.lower().strip().replace("-", " ").replace("_", " ")

def match_symptoms(user_input: str, known_symptoms: list[str]) -> list[str]:
    raw_parts = [p.strip() for p in user_input.replace(" and ", ",").split(",")]
    raw_parts = [p for p in raw_parts if p]
    matched = []
    known_normalized = {normalize(s): s for s in known_symptoms}

    for part in raw_parts:
        part_norm = normalize(part)
        found = None

        for canonical, syns in SYNONYMS.items():
            if part_norm == canonical or part_norm in syns:
                if canonical in known_normalized:
                    found = known_normalized[canonical]
                else:
                    close = difflib.get_close_matches(canonical, known_symptoms, n=1, cutoff=0.7)
                    if close:
                        found = close[0]
                break

        if not found and part_norm in known_normalized:
            found = known_normalized[part_norm]

        if not found:
            close = difflib.get_close_matches(part_norm, list(known_normalized.keys()), n=1, cutoff=0.72)
            if close:
                found = known_normalized[close[0]]

        if not found:
            for known_norm, original in known_normalized.items():
                if part_norm in known_norm or known_norm in part_norm:
                    found = original
                    break

        if found and found not in matched:
            matched.append(found)

    return matched

def symptoms_to_vector(matched_symptoms, all_symptoms):
    return [1 if s in matched_symptoms else 0 for s in all_symptoms]
