import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

st.set_page_config(page_title="Nutrient Deficiency Risk Predictor", layout="wide")

# -----------------------------
# Global styling for PC viewing
# -----------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
            margin: 0 auto;
        }
        h1 { font-size: 2.0rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.2rem !important; }
        html, body, [class*="css"] {
            font-size: 16px;
        }
        .stCaption, .caption {
            font-size: 0.95rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1rem;
            padding: 0.6rem 1rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.95rem !important;
        }
        .stDataFrame, .stTable {
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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

def environmental_recommendations(pred, prob, temperature, humidity, ph, rainfall):
    """Recommendations based on the model's prediction and environmental inputs only.
    Crop is intentionally NOT used here — predictions are crop-agnostic per the study design."""
    recs = []

    if pred == 1:
        recs.append("⚠️ **Deficiency Risk Detected.** The model has 85.8% recall, meaning it correctly identifies approximately 86 out of 100 actual deficiency cases in testing.")
        recs.append("📋 **Suggested actions:** Conduct soil testing to confirm, and apply corrective fertilization based on lab results. Schedule early intervention to reduce yield-loss risk.")
    else:
        recs.append("✅ **Normal Condition Predicted.** Continue routine monitoring and good nutrient management practices.")
        recs.append("📊 The model flags approximately 86% of true deficiency cases, but periodic soil testing is still recommended since some cases are missed.")

    # Environmental cues (these depend on inputs, not crop)
    if rainfall >= 200:
        recs.append("💧 High rainfall (≥200 mm) increases nutrient leaching risk. Consider split fertilizer applications.")
    elif rainfall < 50:
        recs.append("💧 Low rainfall (<50 mm) may slow nutrient release through decomposition. Monitor irrigation and soil moisture.")
    if ph < 5.5:
        recs.append("🧪 Soil is acidic (pH < 5.5). Consider liming based on local recommendations.")
    elif ph > 7.5:
        recs.append("🧪 Soil is alkaline (pH > 7.5). Consider amendments to improve nutrient availability.")
    if humidity >= 85:
        recs.append("💨 High humidity (≥85%) increases plant stress vulnerability. Improve field aeration/drainage where possible.")

    recs.append("🔬 **Top predictors from the study:** Rainfall and humidity were the most important variables for deficiency risk prediction.")
    return recs

def render_centered_pyplot(fig, width_ratio=(1, 2, 1)):
    left, mid, right = st.columns(width_ratio)
    with mid:
        st.pyplot(fig, use_container_width=True)

def make_risk_gauge(prob):
    pct = prob * 100
    band_label, _ = risk_band(prob)

    if pct < 33:
        value_color = "#2e7d32"
    elif pct < 66:
        value_color = "#ef6c00"
    else:
        value_color = "#c62828"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={
            "suffix": "%",
            "font": {"size": 40, "color": value_color, "family": "Arial Black"},
        },
        title={
            "text": f"Deficiency Risk Probability<br><span style='font-size:0.9em;color:#555'>{band_label}</span>",
            "font": {"size": 16},
        },
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555", "tickfont": {"size": 11}},
            "bar": {"color": "rgba(0,0,0,0)"},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#ddd",
            "steps": [
                {"range": [0, 33], "color": "#66bb6a"},
                {"range": [33, 66], "color": "#ffca28"},
                {"range": [66, 100], "color": "#ef5350"},
            ],
            "threshold": {"line": {"color": value_color, "width": 5}, "thickness": 0.85, "value": pct},
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor="white", font={"color": "#333"})
    return fig

# -----------------------------
# Load artifacts
# -----------------------------
model, features = load_model()
results_df, cv_df = load_tables()
cm, fpr, tpr, auc, best_name, thresholds = load_eval_artifacts()

# -----------------------------
# Setup crop data from thresholds (for reference display only)
# -----------------------------
crop_thresholds = thresholds.get("crop_thresholds", {})
crop_list = sorted(crop_thresholds.keys()) if crop_thresholds else []

agronomic_groups = thresholds.get("agronomic_groups", {
    "high_n_demand": ["rice", "maize", "cotton", "jute", "coffee"],
    "legumes": ["chickpea", "kidneybeans", "pigeonpeas", "mothbeans", "mungbean", "blackgram", "lentil"],
    "fruits": ["pomegranate", "banana", "mango", "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya", "coconut"]
})

crop_to_group = {}
for group, crops in agronomic_groups.items():
    for crop in crops:
        crop_to_group[crop] = group

group_display = {
    "high_n_demand": "🔵 High-N-Demand",
    "legumes": "🟢 Legumes",
    "fruits": "🟠 Fruits"
}

# -----------------------------
# UI
# -----------------------------
st.title("🌱 Predictive Analytics: Nutrient Deficiency Risk")
st.caption("Proactive risk estimation using environmental and soil-condition variables (temperature, humidity, pH, rainfall).")

# Global notice about model design
st.info(
    "ℹ️ **How this tool works:** The model predicts deficiency risk from **four environmental variables only** "
    "— temperature, humidity, soil pH, and rainfall. Crop selection (where available) is **reference information** "
    "and does **not** change the prediction. This matches the study's design, which deliberately excludes crop "
    "identity from model inputs to avoid circular prediction."
)

tabs = st.tabs([
    "📊 Predict (Single Input)",
    "🔍 What-if Analysis",
    "📁 Batch Prediction",
    "📈 Model Performance",
    "📖 About / Target Definition"
])

# -----------------------------
# Tab 1: Single prediction
# -----------------------------
with tabs[0]:
    st.subheader("Single Prediction")

    # Environmental inputs come first because they drive the prediction
    st.markdown("### 🌡️ Environmental Conditions *(these drive the prediction)*")
    env1, env2, env3, env4 = st.columns(4)
    with env1:
        temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0, step=0.1)
    with env2:
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0, step=0.1)
    with env3:
        ph = st.number_input("Soil pH", 3.0, 10.0, 6.5, step=0.01)
    with env4:
        rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, 100.0, step=0.1)

    st.markdown("---")

    # Optional crop context — clearly framed as reference only
    st.markdown("### 🌾 Crop Context *(optional — reference only, does not change the prediction)*")
    use_crop = st.checkbox("Show crop-specific reference info", value=False,
                           help="Selecting a crop adds reference NPK thresholds from the study to the results panel. "
                                "It does not change the predicted risk.")

    selected_crop = None
    if use_crop:
        if crop_list:
            crop_col1, crop_col2 = st.columns([1, 2])
            with crop_col1:
                selected_crop = st.selectbox("Reference crop", crop_list)
            with crop_col2:
                crop_group = crop_to_group.get(selected_crop, "unknown")
                st.markdown(f"**Group:** {group_display.get(crop_group, crop_group)}")
                if selected_crop in crop_thresholds:
                    thr = crop_thresholds[selected_crop]
                    st.caption(
                        f"📊 Study thresholds (25th percentile) for {selected_crop}: "
                        f"N < {thr['N_thr']}, P < {thr['P_thr']}, K < {thr['K_thr']} — "
                        f"shown for reference only."
                    )
        else:
            st.warning("No crop thresholds found. Check thresholds.json.")

    st.markdown("---")

    # Predict button
    if st.button("🔮 Predict Risk", type="primary"):
        pred, prob = predict_one(model, features, temperature, humidity, ph, rainfall)
        band, color = risk_band(prob if prob is not None else 0.0)

        st.markdown("### 📋 Prediction Result")
        st.caption("Based on environmental inputs only. Crop selection (if any) does not affect this result.")

        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            if pred == 1:
                st.error("⚠️ **DEFICIENCY RISK DETECTED**")
                st.metric("Predicted Class", "Deficiency Risk (1)")
            else:
                st.success("✅ **NORMAL CONDITION**")
                st.metric("Predicted Class", "Normal (0)")
            st.metric("Risk Probability", f"{prob*100:.2f}%" if prob is not None else "N/A")
            st.markdown(f"**Risk Band:** :{color}[{band}]")

        with col_res2:
            if prob is not None:
                gauge_fig = make_risk_gauge(prob)
                st.plotly_chart(gauge_fig, use_container_width=True)

        # Reference crop context panel (only if user opted in)
        if use_crop and selected_crop and selected_crop in crop_thresholds:
            thr = crop_thresholds[selected_crop]
            crop_group = crop_to_group.get(selected_crop, "unknown")
            st.markdown("### 🌾 Crop Reference Panel")
            st.caption("Informational only — these values were used during model training to construct the label, "
                       "not as inputs to the prediction above.")
            st.markdown(
                f"- **Reference crop:** {selected_crop}\n"
                f"- **Agronomic group:** {group_display.get(crop_group, crop_group)}\n"
                f"- **Crop-specific 25th-percentile NPK thresholds (from training set):** "
                f"N < {thr['N_thr']}, P < {thr['P_thr']}, K < {thr['K_thr']}\n"
                f"- **Interpretation:** In the study, a row for {selected_crop} was labeled "
                f"*Deficiency Risk* if any of its N, P, or K values fell below these thresholds "
                f"(Liebig's law of the minimum)."
            )

        st.markdown("### 📋 Recommendations")
        recs = environmental_recommendations(pred, prob, temperature, humidity, ph, rainfall)
        for r in recs:
            st.write(r)

        # Download report
        report = pd.DataFrame([{
            "reference_crop": selected_crop if selected_crop else "not_selected",
            "reference_crop_group": crop_to_group.get(selected_crop, "not_selected") if selected_crop else "not_selected",
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
            "predicted_class": pred,
            "risk_probability": prob,
            "note": "Prediction uses environmental variables only; reference_crop is informational."
        }])
        st.download_button(
            "📥 Download Prediction Report (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name=f"prediction_{selected_crop if selected_crop else 'report'}.csv",
            mime="text/csv"
        )

# -----------------------------
# Tab 2: What-if analysis
# -----------------------------
with tabs[1]:
    st.subheader("What-if Analysis (Sensitivity)")
    st.write("Adjust one variable while holding the others constant to see how predicted risk changes.")
    st.caption("This sweep uses only environmental variables — crop is not a model input.")

    base_temp = st.slider("Base Temperature (°C)", 0.0, 60.0, 25.0, 0.5)
    base_hum = st.slider("Base Humidity (%)", 0.0, 100.0, 70.0, 1.0)
    base_ph = st.slider("Base Soil pH", 3.0, 10.0, 6.5, 0.05)
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

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=110)
    ax.plot(xs, np.array(probs)*100, linewidth=2, color="#2e7d32")
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel("Deficiency Risk Probability (%)", fontsize=10)
    ax.set_title(f"What-if Curve: Risk vs {var.capitalize()}", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    render_centered_pyplot(fig, width_ratio=(1, 3, 1))

# -----------------------------
# Tab 3: Batch prediction
# -----------------------------
with tabs[2]:
    st.subheader("Batch Prediction (CSV Upload)")
    st.write("Upload a CSV with columns: **temperature, humidity, ph, rainfall**")
    st.caption("Crop columns in the uploaded file will be passed through unchanged — they do not affect the prediction.")

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
            out["risk_band"] = out["risk_probability"].apply(lambda x: risk_band(x)[0] if not np.isnan(x) else "N/A")

            st.dataframe(out.head(20), use_container_width=True)

            st.download_button(
                "📥 Download Predictions (CSV)",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="batch_predictions.csv",
                mime="text/csv"
            )

# -----------------------------
# Tab 4: Performance page
# -----------------------------
with tabs[3]:
    st.subheader("Model Performance (RQ2 & RQ3)")
    st.write(f"**Best model:** `{best_name}` | **ROC-AUC:** `{auc:.3f}` | **Recall:** `0.858` (85.8%)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Test-set Comparison**")
        st.dataframe(results_df.sort_values("F1", ascending=False), use_container_width=True)
    with c2:
        st.markdown("**10-fold Cross-Validation**")
        st.dataframe(cv_df.sort_values("CV_F1_Mean", ascending=False), use_container_width=True)

    st.markdown("### Confusion Matrix (Best Model)")
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=110)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal (0)", "Deficiency Risk (1)"],
                yticklabels=["Normal (0)", "Deficiency Risk (1)"],
                annot_kws={"size": 10}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(f"Confusion Matrix - {best_name}", fontsize=11)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    render_centered_pyplot(fig, width_ratio=(1, 2, 1))

    st.markdown("### ROC Curve (Best Model)")
    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=110)
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2, color="#2e7d32")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"ROC Curve - {best_name}", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    render_centered_pyplot(fig, width_ratio=(1, 2, 1))

# -----------------------------
# Tab 5: About/target definition
# -----------------------------
with tabs[4]:
    st.subheader("About / Target Definition")

    st.markdown("""
    ### Why the crop dropdown does not change the prediction

    This is the most common point of confusion, so it is worth stating directly:

    - The model was trained on **four environmental features only**: temperature, humidity, soil pH, and rainfall.
    - **Crop identity was deliberately excluded** from the model's inputs to prevent circular prediction.
      The crop label was used **only** to construct the training label (the deficiency flag), not as a feature.
    - The "Crop Context" selector in the Predict tab is therefore **informational**. It pulls up the
      crop-specific NPK thresholds that were used during label construction, so you can see what
      "Deficiency Risk" means in agronomic terms — but it does not change the model's output.

    Selecting "rice" or "apple" or "lentil" with the same temperature, humidity, pH, and rainfall
    will produce **the same predicted risk probability**. The crop panel only changes the *reference
    information* shown alongside the result.

    ### How the Deficiency Risk Label Was Constructed

    The prototype predicts a binary **Nutrient Deficiency Risk** label derived from soil Nitrogen (N),
    Phosphorus (P), and Potassium (K) values — but **N, P, K are NOT used as model inputs**.

    **Target Rule (Liebig's Law of the Minimum):**
    > A record is labeled **Deficiency Risk (1)** if **ANY** of N, P, or K falls below its
    > **crop-specific 25th percentile threshold** computed from the training set.

    This OR-based rule reflects the agronomic principle that crop growth is constrained by the
    **most limiting macronutrient** (Brady & Weil, 2017; Marschner, 2012).

    ### Why crop-specific thresholds (and grouping) mattered during training

    Even though crop is not a model input, the way the label was *built* is crop-aware. A single
    global 25th-percentile threshold would have unfairly flagged ~64% of fruits but only ~31% of
    high-N crops, because cereals naturally have high N and fruits naturally have high K. The
    crop-specific approach balanced flag rates across agronomic groups (52.9–56.3%), producing a
    cleaner training signal. The grouping itself (high-N-demand / legumes / fruits) was the
    organizing principle for that label construction.

    ### Expected vs. Observed Flag Rates

    With three independent 25th-percentile thresholds, the theoretical probability of being flagged is:

    1 - (0.75)³ = 0.578 (57.8%)

    **Observed flag rates in this dataset:**
    - Fruits: 52.9%
    - High-N-Demand crops: 56.3%
    - Legumes: 54.0%

    The close match confirms the OR rule behaves as expected, with slight deviation due to weak
    positive correlations among N, P, and K.

    ### Crop-Specific Thresholds (25th Percentile Examples)

    | Class | Crops | Example Thresholds (N, P, K) |
    | :---- | :---- | :--------------------------- |
    | **High-N-Demand** | rice, maize, cotton, jute, coffee | Rice: (69, 40, 38) |
    | **Legumes** | chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil | Chickpea: (31, 62, 77) |
    | **Fruits** | pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut | Pomegranate: (7, 13, 38) |

    ### Predictors Used by the Model

    | Variable | Unit | Range in Dataset |
    | :------- | :--- | :--------------- |
    | Temperature | °C | 8.8 - 43.7 |
    | Humidity | % | 14.3 - 100.0 |
    | Soil pH | unitless | 3.5 - 9.9 |
    | Rainfall | mm | 20.2 - 298.6 |

    ### Model Performance Summary

    - **Best Model:** Random Forest
    - **Test Accuracy:** 76.1%
    - **Recall (Deficiency Risk):** 85.8% — correctly flags ~86 of 100 actual deficiency cases
    - **ROC-AUC:** 0.854 — strong discriminative ability
    - **Top Predictors:** Rainfall (0.301), Humidity (0.291)

    > ⚠️ **Limitation:** The label is a distribution-derived proxy, not a field-validated diagnosis.
    > Use this tool as a screening aid to prioritize soil testing, not as a replacement for
    > professional agronomic advice.
    """)

    with st.expander("📁 View Complete Crop-Specific Thresholds (JSON)"):
        st.json(thresholds)