from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from safety import DRUG_INFO, safety_review

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "drug_recommender.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

st.set_page_config(
    page_title="ML Drug Recommendation Demo",
    page_icon="💊",
    layout="wide",
)

st.title("💊 Machine Learning Drug Recommendation System")
st.warning(
    "Educational prototype only."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text())

model = load_model()
metrics = load_metrics()

if model is None:
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()

with st.sidebar:
    st.header("Patient Information")
    age = st.slider("Age", 18, 90, 45)
    sex = st.selectbox("Sex", ["Female", "Male"])
    condition = st.selectbox(
        "Primary condition",
        ["Hypertension", "Type 2 Diabetes", "High Cholesterol", "Asthma", "Acid Reflux"],
    )
    bmi = st.number_input("BMI", 12.0, 60.0, 26.0, step=0.1)
    systolic_bp = st.number_input("Systolic blood pressure", 70, 250, 125)
    glucose = st.number_input("Glucose (mg/dL)", 40, 500, 105)
    cholesterol = st.number_input("Total cholesterol (mg/dL)", 80, 500, 190)
    kidney_disease = st.checkbox("Kidney disease")
    liver_disease = st.checkbox("Liver disease")
    allergy_penicillin = st.checkbox("Penicillin allergy")
    pregnant = st.checkbox("Pregnant", disabled=(sex == "Male"))
    recommend = st.button("Generate recommendation", type="primary", use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Recommendation", "Model Performance", "About"])

with tab1:
    if recommend:
        patient = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "glucose": glucose,
            "cholesterol": cholesterol,
            "kidney_disease": int(kidney_disease),
            "liver_disease": int(liver_disease),
            "allergy_penicillin": int(allergy_penicillin),
            "pregnant": int(pregnant if sex == "Female" else False),
            "condition": condition,
        }

        input_df = pd.DataFrame([patient])
        predicted_drug = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_

        probability_df = (
            pd.DataFrame({"Drug": classes, "Probability": probabilities})
            .sort_values("Probability", ascending=False)
            .head(5)
        )

        top_confidence = float(probability_df.iloc[0]["Probability"])
        warnings = safety_review(predicted_drug, patient)
        info = DRUG_INFO.get(predicted_drug, {})

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Model output")
            st.success(f"Suggested class: **{predicted_drug}**")
            st.metric("Model confidence", f"{top_confidence:.1%}")
            st.write(f"**Purpose:** {info.get('purpose', 'Not available')}")
            st.write(
                f"**Common side effects:** "
                f"{info.get('common_side_effects', 'Not available')}"
            )

            if top_confidence < 0.55:
                st.error(
                    "Low model confidence. A qualified clinician should review the case."
                )

            if warnings:
                st.subheader("Safety review")
                for warning in warnings:
                    st.error(warning)
            else:
                st.info(
                    "No rule-based warning was triggered, but a clinician must still "
                    "review allergies, interactions, dosage, laboratory values, and history."
                )

        with col2:
            st.subheader("Top model probabilities")
            fig = px.bar(
                probability_df,
                x="Probability",
                y="Drug",
                orientation="h",
                text=probability_df["Probability"].map(lambda x: f"{x:.1%}"),
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Entered patient profile")
        st.dataframe(input_df, use_container_width=True)

    else:
        st.info("Enter patient information in the sidebar and select Generate recommendation.")

with tab2:
    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
        col2.metric("Training rows", metrics["training_rows"])
        col3.metric("Test rows", metrics["test_rows"])

        report = pd.DataFrame(metrics["classification_report"]).T
        st.dataframe(report.round(3), use_container_width=True)
        st.caption(
            "Performance is measured on a synthetic demonstration dataset and must not "
            "be interpreted as clinical validation."
        )
    else:
        st.info("Metrics will appear after model training.")

with tab3:
    st.markdown(
        """
        ### Project workflow

        1. A synthetic patient dataset is loaded.
        2. Categorical variables are one-hot encoded.
        3. A Random Forest classifier predicts a medication class.
        4. Prediction probabilities are displayed.
        5. Rule-based checks flag selected high-risk situations.
        6. All outputs remain subject to licensed clinician review.

        ### Important limitations

        The dataset is synthetic and simplified. A production clinical system would require
        validated medical data, medication interaction databases, dosage logic, subgroup
        analysis, external validation, privacy controls, audit trails, regulatory review,
        human oversight, and continuous safety monitoring.
        """
    )
