import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Nutrient Deficiency Risk Predictor", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("artifacts/best_model.pkl")
    features = joblib.load("artifacts/feature_list.pkl")
    return model, features

@st.cache_data
def load_tables():
    results_df = pd.read_csv("artifacts/results_df.csv")
    cv_df = pd.read_csv("artifacts/cv_df.csv")
    return results_df, cv_df

@st.cache_data
def load_eval_artifacts():
    cm = np.load("artifacts/confusion_matrix.npy")
    fpr = np.load("artifacts/roc_fpr.npy")
    tpr = np.load("artifacts/roc_tpr.npy")
    with open("artifacts/roc_auc.txt", "r") as f:
        auc = float(f.read().strip())
    with open("artifacts/best_model_name.txt", "r") as f:
        best_name = f.read().strip()
    with open("artifacts/thresholds.json", "r") as f:
        thresholds = json.load(f)
    return cm, fpr, tpr, auc, best_name, thresholds

def risk_band(p):
    if p < 0.33:
        return "LOW", "green"
    elif p < 0.66:
        return "MODERATE", "orange"
    return "HIGH", "red"

def predict_one(model, features, temperature, humidity, ph, rainfall):
    X = pd.DataFrame([[temperature, humidity, ph, rainfall]], columns=features)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None
    return pred, prob

def recommendation_text(pred, prob, temperature, humidity, ph, rainfall):
    recs = []
    if pred == 1:
        recs.append("Consider conducting a soil test and applying corrective fertilization based on nutrient results.")
        recs.append("Monitor crop condition closely and schedule early intervention to reduce yield loss risk.")
    else:
        recs.append("Conditions appear lower-risk. Continue routine monitoring and good nutrient management practices.")

    if rainfall >= 200:
        recs.append("High rainfall can increase nutrient leaching; consider split fertilizer applications and monitoring.")
    if ph < 5.5:
        recs.append("Soil is acidic (pH < 5.5). Consider liming based on local recommendations.")
    if ph > 7.5:
        recs.append("Soil is alkaline (pH > 7.5). Consider amendments to improve nutrient availability based on local guidance.")
    if humidity >= 85:
        recs.append("High humidity can increase plant stress vulnerability; improve field aeration/drainage if applicable.")
    return recs

# -----------------------------
# Load artifacts
# -----------------------------
model, features = load_model()
results_df, cv_df = load_tables()
cm, fpr, tpr, auc, best_name, thresholds = load_eval_artifacts()

# -----------------------------
# UI
# -----------------------------
st.title("Predictive Analytics: Nutrient Deficiency Risk")
st.caption("Proactive risk estimation using environmental and soil-condition variables (temperature, humidity, pH, rainfall).")

tabs = st.tabs([
    "Predict (Single Input)",
    "What-if Analysis",
    "Batch Prediction",
    "Model Performance",
    "About / Target Definition"
])

# -----------------------------
# Tab 1: Single prediction
# -----------------------------
with tabs[0]:
    st.subheader("Single Prediction")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0, step=0.1)
    with c2:
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0, step=0.1)
    with c3:
        ph = st.number_input("Soil pH", 3.0, 10.0, 6.5, step=0.01)
    with c4:
        rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, 100.0, step=0.1)

    if st.button("Predict Risk"):
        pred, prob = predict_one(model, features, temperature, humidity, ph, rainfall)
        band, color = risk_band(prob if prob is not None else 0.0)

        st.markdown("### Result")
        colA, colB, colC = st.columns(3)
        colA.metric("Predicted Class", "Deficiency Risk (1)" if pred == 1 else "Normal (0)")
        colB.metric("Risk Probability (Class 1)", f"{prob*100:.2f}%" if prob is not None else "N/A")
        colC.markdown(f"**Risk Band:** :{color}[{band}]")

        st.markdown("### Recommendations")
        recs = recommendation_text(pred, prob, temperature, humidity, ph, rainfall)
        for r in recs:
            st.write("- " + r)

        # Download small report
        report = pd.DataFrame([{
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
            "predicted_class": pred,
            "risk_probability": prob
        }])
        st.download_button(
            "Download Prediction Report (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="prediction_report.csv",
            mime="text/csv"
        )

# -----------------------------
# Tab 2: What-if analysis
# -----------------------------
with tabs[1]:
    st.subheader("What-if Analysis (Sensitivity)")
    st.write("Adjust one variable while holding the others constant to see how predicted risk changes.")

    base_temp = st.slider("Base Temperature (°C)", 0.0, 60.0, 25.0, 0.5)
    base_hum  = st.slider("Base Humidity (%)", 0.0, 100.0, 70.0, 1.0)
    base_ph   = st.slider("Base Soil pH", 3.0, 10.0, 6.5, 0.05)
    base_rain = st.slider("Base Rainfall (mm)", 0.0, 400.0, 100.0, 5.0)

    var = st.selectbox("Variable to vary", ["temperature", "humidity", "ph", "rainfall"])

    if var == "temperature":
        xs = np.linspace(0, 60, 61)
        probs = [predict_one(model, features, x, base_hum, base_ph, base_rain)[1] for x in xs]
        xlab = "Temperature (°C)"
    elif var == "humidity":
        xs = np.linspace(0, 100, 101)
        probs = [predict_one(model, features, base_temp, x, base_ph, base_rain)[1] for x in xs]
        xlab = "Humidity (%)"
    elif var == "ph":
        xs = np.linspace(3, 10, 71)
        probs = [predict_one(model, features, base_temp, base_hum, x, base_rain)[1] for x in xs]
        xlab = "Soil pH"
    else:
        xs = np.linspace(0, 400, 81)
        probs = [predict_one(model, features, base_temp, base_hum, base_ph, x)[1] for x in xs]
        xlab = "Rainfall (mm)"

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(xs, np.array(probs)*100)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Deficiency Risk Probability (%)")
    ax.set_title(f"What-if Curve: Risk vs {var}")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# -----------------------------
# Tab 3: Batch prediction
# -----------------------------
with tabs[2]:
    st.subheader("Batch Prediction (CSV Upload)")
    st.write("Upload a CSV with columns: temperature, humidity, ph, rainfall")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        data = pd.read_csv(file)
        missing = [c for c in features if c not in data.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            X = data[features].copy()
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.nan

            out = data.copy()
            out["predicted_class"] = preds
            out["risk_probability"] = probs

            st.dataframe(out.head(20))

            st.download_button(
                "Download Predictions (CSV)",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="batch_predictions.csv",
                mime="text/csv"
            )

# -----------------------------
# Tab 4: Performance page
# -----------------------------
with tabs[3]:
    st.subheader("Model Performance (RQ2 & RQ3)")
    st.write(f"**Best model:** {best_name} | **ROC-AUC:** {auc:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Test-set Comparison (results_df)**")
        st.dataframe(results_df.sort_values("F1", ascending=False))
    with c2:
        st.markdown("**10-fold Cross-Validation (cv_df)**")
        st.dataframe(cv_df.sort_values("CV_F1_Mean", ascending=False))

    st.markdown("### Confusion Matrix (Best Model)")
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal (0)", "Deficiency Risk (1)"],
                yticklabels=["Normal (0)", "Deficiency Risk (1)"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {best_name}")
    st.pyplot(fig)

    st.markdown("### ROC Curve (Best Model)")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0,1], [0,1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {best_name}")
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# Tab 5: About/target definition
# -----------------------------
with tabs[4]:
    st.subheader("About / Target Definition")
    st.write("This prototype predicts a binary nutrient deficiency risk label derived from soil N, P, and K values.")
    st.json(thresholds)
    st.write(
        "Target rule: `Deficiency Risk = 1` if any of N, P, or K is below the 25th percentile thresholds "
        "(computed from the training split); otherwise `Normal = 0`."
    )
    st.write(
        "Predictors used by the model: temperature, humidity, soil pH, rainfall. "
        "This avoids circular prediction because the target was derived from N/P/K."
    )