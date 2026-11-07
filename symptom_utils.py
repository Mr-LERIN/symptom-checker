import difflib
import re
import pandas as pd

# Keys MUST match actual dataset column names (with underscores)
SYNONYMS = {
    "high_fever": ["fever", "high temperature", "temperature", "pyrexia", "hot", "burning up"],
    "headache": ["head pain", "migraine", "head ache"],
    "nausea": ["feeling sick", "queasy", "nauseous", "want to vomit"],
    "vomiting": ["throwing up", "puking", "emesis", "barfing"],
    "fatigue": ["tiredness", "exhaustion", "feeling tired", "tired", "lethargic", "no energy"],
    "cough": ["coughing", "hacking", "chesty cough"],
    "sore_throat": ["throat pain", "painful throat", "scratchy throat", "throat hurts"],
    "runny_nose": ["rhinorrhea", "nose running", "dripping nose", "snotty nose"],
    "breathlessness": ["shortness of breath", "difficulty breathing", "dyspnea", "cant breathe", "hard to breathe"],
    "chest_pain": ["pain in chest", "chest ache", "tight chest"],
    "stomach_pain": ["abdominal pain", "belly pain", "tummy pain", "stomach ache", "gut pain"],
    "diarrhoea": ["loose stools", "diarrhea", "the runs", "watery stool"],
    "constipation": ["difficulty passing stool", "cant poop", "blocked up"],
    "dizziness": ["lightheaded", "giddiness", "spinning", "vertigo", "room spinning"],
    "skin_rash": ["rash", "skin eruption", "dermatitis", "red skin"],
    "itching": ["itchy", "pruritus", "itch"],
    "yellowish_skin": ["yellow skin", "jaundice skin", "yellow complexion"],
    "yellowing_of_eyes": ["yellow eyes", "jaundice eyes", "yellow sclera"],
    "loss_of_appetite": ["not hungry", "no appetite", "anorexia", "cant eat"],
    "excessive_hunger": ["hungry all the time", "always hungry", "polyphagia", "cant stop eating"],
    "polyuria": ["peeing a lot", "frequent urination", "pee often", "urinating frequently"],
    "blurred_and_distorted_vision": ["blurred vision", "vision blurry", "cant see clear", "fuzzy vision"],
    "back_pain": ["back ache", "sore back", "spinal pain"],
    "joint_pain": ["arthralgia", "aching joints", "joints hurt"],
    "muscle_pain": ["myalgia", "aching muscles", "body aches", "muscle ache", "muscles sore"],
    "neck_pain": ["sore neck", "neck ache", "stiff neck", "nuchal pain"],
    "chills": ["cold sweats", "shivering", "rigors", "freezing"],
    "sweating": ["perspiration", "diaphoresis", "sweating a lot", "clammy"],
    "dehydration": ["dry mouth", "thirsty", "dehydrated", "parched"],
    "indigestion": ["dyspepsia", "upset stomach", "heartburn", "acid reflux"],
    "anxiety": ["anxious", "nervous", "worry", "panic", "stressed"],
    "depression": ["depressed", "sad", "feeling down", "low mood", "hopeless"],
    "irritability": ["irritable", "moody", "short tempered", "snappy"],
    "weight_loss": ["losing weight", "slimming down", "unintended weight loss", "wasting"],
    "weight_gain": ["gaining weight", "putting on weight", "swelling up"],
    "fast_heart_rate": ["tachycardia", "racing heart", "heart pounding", "palpitations"],
    "swelled_lymph_nodes": ["swollen glands", "lymphadenopathy", "lumps in neck", "neck lumps"],
    "malaise": ["feeling unwell", "general discomfort", "under the weather", "not right"],
    "phlegm": ["mucus", "sputum", "chest congestion", "chesty"],
    "throat_irritation": ["irritated throat", "tickle in throat", "throat scratchy"],
    "redness_of_eyes": ["red eyes", "bloodshot eyes", "pink eye"],
    "sinus_pressure": ["sinus pain", "facial pressure", "cheek pain"],
    "congestion": ["stuffy nose", "nasal congestion", "blocked nose"],
    "bruising": ["bruises", "ecchymosis", "black and blue", "contusion"],
    "obesity": ["overweight", "fat", "heavy", "obese"],
    "swollen_legs": ["leg swelling", "edema legs", "puffy legs", "water in legs"],
    "swollen_blood_vessels": ["prominent veins", "varicose veins", "spider veins"],
    "puffy_face_and_eyes": ["facial swelling", "swollen eyes", "puffy face", "bags under eyes"],
    "enlarged_thyroid": ["goiter", "thyroid swelling", "neck swelling front"],
    "brittle_nails": ["nail breakage", "weak nails", "crumbling nails"],
    "swollen_extremeties": ["swollen hands", "swollen feet", "puffy extremities", "fingers swollen"],
    "cold_hands_and_feets": ["cold hands", "cold feet", "circulation poor", "icy fingers"],
    "burning_micturition": ["burning pee", "painful urination", "dysuria", "pee stings"],
    "spotting_urination": ["blood in urine", "hematuria", "spots in pee", "pink urine"],
    "dark_urine": ["brown urine", "tea colored urine", "cola urine"],
    "foul_smell_of_urine": ["smelly urine", "urine odor", "stinky pee"],
    "continuous_feel_of_urine": ["constant urge to pee", "overactive bladder", "cant hold pee"],
    "passage_of_gases": ["gas", "flatulence", "farting", "bloating", "wind"],
    "internal_itching": ["itch inside", "internal itch", "deep itch"],
    "toxic_look_typhos": ["typhoid look", "toxic appearance", "very ill looking", "typhoid face"],
    "belly_pain": ["belly ache", "tummy ache", "navel pain"],
    "abnormal_menstruation": ["irregular periods", "heavy periods", "menstrual issues", "period problems"],
    "dischromic_patches": ["skin patches", "discolored skin", "hypopigmentation", "vitiligo spots"],
    "watering_from_eyes": ["tearing", "watery eyes", "eye discharge", "eyes watering"],
    "increased_appetite": ["hungry more", "eating more", "ravenous"],
    "family_history": ["runs in family", "genetic", "hereditary", "family had it"],
    "mucoid_sputum": ["clear phlegm", "mucus white", "sticky phlegm"],
    "rusty_sputum": ["bloody phlegm", "rust colored sputum", "brown phlegm"],
    "blood_in_sputum": ["coughing blood", "hemoptysis", "blood in spit"],
    "pain_during_bowel_movements": ["painful poop", "painful defecation", "hurts to go"],
    "pain_in_anal_region": ["anal pain", "rectal pain", "hemorrhoid pain", "pain in bottom"],
    "bloody_stool": ["blood in stool", "rectal bleeding", "melena", "red stool"],
    "irritation_in_anus": ["anal itching", "pruritus ani", "itchy bottom"],
    "acute_liver_failure": ["liver failure", "hepatic failure", "yellow skin and eyes"],
    "fluid_overload": ["water retention", "edema", "swelling from fluid", "anasarca"],
    "swelling_of_stomach": ["bloated stomach", "distended abdomen", "ascites", "big belly"],
    "history_of_alcohol_consumption": ["alcoholic", "drinks alcohol", "heavy drinking", "alcohol abuse"],
    "prominent_veins_on_calf": ["calf veins", "leg veins visible", "veins popping out legs"],
    "stiff_neck": ["neck stiffness", "nuchal rigidity", "cant move neck", "neck locked"],
    "loss_of_balance": ["unsteady", "falling over", "balance problems", "ataxia"],
    "unsteadiness": ["wobbly", "unstable gait", "stumbling"],
    "weakness_of_one_body_side": ["hemiparesis", "one side weak", "paralysis one side", "left side weak", "right side weak"],
    "altered_sensorium": ["confusion", "altered mental status", "drowsiness", "coma", "lethargy", "not alert"],
    "bladder_discomfort": ["bladder pain", "urinary discomfort", "pain over bladder"],
    "receiving_blood_transfusion": ["blood transfusion", "transfused", "given blood"],
    "receiving_unsterile_injections": ["dirty needle", "unsterile injection", "shared needle"],
    "coma": ["unconscious", "passed out", "unresponsive", "knocked out"],
    "stomach_bleeding": ["gi bleed", "gastrointestinal bleeding", "vomiting blood", "blood in vomit"],
    "distention_of_abdomen": ["abdomen distended", "bloated belly", "stomach swollen"],
    "palpitations": ["heart palpitations", "skipped beat", "fluttering heart", "heart flip flops"],
    "painful_walking": ["limping", "gait pain", "hurts to walk", "cant walk properly"],
    "pus_filled_pimples": ["pustules", "acne", "zits", "whiteheads", "pimples with pus"],
    "blackheads": ["comedones", "clogged pores", "open comedones"],
    "scurring": ["scarring", "acne scars", "pock marks"],
    "silver_like_dusting": ["silver scales", "psoriasis scales", "fish scale skin"],
    "small_dents_in_nails": ["nail pitting", "dented nails", "ice pick nails"],
    "inflammatory_nails": ["nail inflammation", "paronychia", "red nails", "swollen nail fold"],
    "red_sore_around_nose": ["nasal sore", "nose infection", "impetigo nose", "crusty nose"],
    "yellow_crust_ooze": ["honey colored crust", "weeping skin", "golden crust"],
}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def match_symptoms(user_input: str, known_symptoms: list[str]) -> list[str]:
    raw_parts = [p.strip() for p in user_input.replace(" and ", ",").split(",")]
    raw_parts = [p for p in raw_parts if p]
    matched = []
    known_normalized = {normalize(s): s for s in known_symptoms}

    for part in raw_parts:
        part_norm = normalize(part)
        found = None

        # 1. Synonym lookup
        for canonical_name, syns in SYNONYMS.items():
            syns_norm = [normalize(s) for s in syns] + [normalize(canonical_name)]
            if part_norm in syns_norm:
                if canonical_name in known_symptoms:
                    found = canonical_name
                elif normalize(canonical_name) in known_normalized:
                    found = known_normalized[normalize(canonical_name)]
                break

        # 2. Exact match
        if not found and part_norm in known_normalized:
            found = known_normalized[part_norm]

        # 3. Fuzzy match
        if not found:
            close = difflib.get_close_matches(
                part_norm, list(known_normalized.keys()), n=1, cutoff=0.72
            )
            if close:
                found = known_normalized[close[0]]

        # 4. Substring fallback
        if not found:
            for known_norm, original in known_normalized.items():
                if part_norm in known_norm or known_norm in part_norm:
                    found = original
                    break

        if found and found not in matched:
            matched.append(found)

    return matched


def symptoms_to_vector(matched_symptoms: list[str], symptom_list: list[str]):
    """
    THE BRIDGE: Returns a pandas DataFrame with named columns.
    This silences the sklearn feature-names warning.
    """
    vec = pd.Series(0, index=symptom_list)
    for s in matched_symptoms:
        if s in vec.index:
            vec[s] = 1
    return pd.DataFrame([vec.values], columns=symptom_list)