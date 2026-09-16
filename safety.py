from typing import Dict, List

DRUG_INFO = {
    "Lisinopril": {
        "purpose": "ACE inhibitor commonly used for hypertension.",
        "common_side_effects": "Dry cough, dizziness, increased potassium.",
    },
    "Amlodipine": {
        "purpose": "Calcium-channel blocker commonly used for hypertension.",
        "common_side_effects": "Ankle swelling, flushing, dizziness.",
    },
    "Losartan": {
        "purpose": "ARB commonly used for hypertension.",
        "common_side_effects": "Dizziness, increased potassium.",
    },
    "Metformin": {
        "purpose": "First-line medicine commonly used for type 2 diabetes.",
        "common_side_effects": "Nausea, diarrhea, abdominal discomfort.",
    },
    "Glipizide": {
        "purpose": "Sulfonylurea used to lower blood glucose.",
        "common_side_effects": "Low blood sugar, weight gain.",
    },
    "Sitagliptin": {
        "purpose": "DPP-4 inhibitor used for type 2 diabetes.",
        "common_side_effects": "Headache, upper respiratory symptoms.",
    },
    "Atorvastatin": {
        "purpose": "Statin used to lower LDL cholesterol.",
        "common_side_effects": "Muscle pain, elevated liver enzymes.",
    },
    "Rosuvastatin": {
        "purpose": "High-potency statin used to lower LDL cholesterol.",
        "common_side_effects": "Muscle pain, headache.",
    },
    "Simvastatin": {
        "purpose": "Statin used to lower LDL cholesterol.",
        "common_side_effects": "Muscle pain, digestive discomfort.",
    },
    "Albuterol": {
        "purpose": "Short-acting bronchodilator used for quick asthma relief.",
        "common_side_effects": "Tremor, fast heartbeat, nervousness.",
    },
    "Budesonide": {
        "purpose": "Inhaled corticosteroid used for asthma control.",
        "common_side_effects": "Oral thrush, hoarseness.",
    },
    "Montelukast": {
        "purpose": "Leukotriene receptor antagonist used for asthma.",
        "common_side_effects": "Headache; mood or behavior changes require prompt review.",
    },
    "Omeprazole": {
        "purpose": "Proton-pump inhibitor used for acid reflux.",
        "common_side_effects": "Headache, abdominal discomfort.",
    },
    "Pantoprazole": {
        "purpose": "Proton-pump inhibitor used for acid reflux.",
        "common_side_effects": "Headache, diarrhea.",
    },
    "Famotidine": {
        "purpose": "H2 blocker used to reduce stomach acid.",
        "common_side_effects": "Headache, dizziness, constipation.",
    },
}

def safety_review(drug: str, patient: Dict) -> List[str]:
    warnings: List[str] = []

    if patient.get("pregnant"):
        if drug in {"Lisinopril", "Losartan", "Atorvastatin", "Rosuvastatin", "Simvastatin"}:
            warnings.append(
                f"{drug} may be inappropriate during pregnancy. Immediate clinician review is required."
            )

    if patient.get("kidney_disease"):
        if drug == "Metformin":
            warnings.append(
                "Kidney function must be assessed before metformin use because dosing or avoidance may be necessary."
            )
        if drug in {"Lisinopril", "Losartan", "Sitagliptin", "Famotidine"}:
            warnings.append(
                f"{drug} may require kidney-function monitoring or dose adjustment."
            )

    if patient.get("liver_disease"):
        if drug in {"Atorvastatin", "Rosuvastatin", "Simvastatin"}:
            warnings.append(
                "Statin therapy requires clinician assessment and liver monitoring in patients with liver disease."
            )

    if patient.get("age", 0) >= 65:
        warnings.append(
            "Older adults may require lower starting doses, interaction review, and closer monitoring."
        )

    if patient.get("systolic_bp", 120) >= 180:
        warnings.append(
            "Systolic blood pressure at or above 180 mmHg may require urgent medical evaluation."
        )

    if patient.get("glucose", 100) >= 250:
        warnings.append(
            "Very high glucose may require prompt clinical assessment rather than routine recommendation."
        )

    return warnings
