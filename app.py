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

def recommendation_text(pred, prob, temperature, humidity, ph, rainfall, selected_crop=None, crop_thresholds=None):
    recs = []
    
    # Recall-based context from manuscript (85.8% recall)
    if pred == 1:
        recs.append("⚠️ **Deficiency Risk Detected.** This model has 85.8% recall, meaning it correctly identifies approximately 86 out of 100 actual deficiency cases.")
        if selected_crop:
            recs.append(f"📋 **Recommended actions for {selected_crop.upper()}:** Conduct soil testing and apply corrective fertilization based on results. Schedule early intervention to reduce yield loss risk.")
        else:
            recs.append("📋 **Recommended actions:** Conduct soil testing and apply corrective fertilization based on results. Schedule early intervention to reduce yield loss risk.")
    else:
        recs.append("✅ **Normal Condition Predicted.** Continue routine monitoring and good nutrient management practices.")
        recs.append("📊 The model flags approximately 86% of true deficiency cases, so periodic soil testing is still recommended.")
    
    # Show crop-specific thresholds if available
    if selected_crop and crop_thresholds and selected_crop in crop_thresholds:
        thr = crop_thresholds[selected_crop]
        recs.append(f"🔬 **{selected_crop.upper()} deficiency thresholds (25th percentile):** N < {thr['N_thr']}, P < {thr['P_thr']}, K < {thr['K_thr']}")
    
    # Environmental recommendations
    if rainfall >= 200:
        recs.append("💧 High rainfall (>200mm) increases nutrient leaching risk. Consider split fertilizer applications.")
    if ph < 5.5:
        recs.append("🧪 Soil is acidic (pH < 5.5). Consider liming based on local recommendations.")
    if ph > 7.5:
        recs.append("🧪 Soil is alkaline (pH > 7.5). Consider amendments to improve nutrient availability.")
    if humidity >= 85:
        recs.append("💨 High humidity (≥85%) increases plant stress vulnerability. Improve field aeration/drainage if applicable.")
    
    # Feature importance context from manuscript
    recs.append("🔬 **Key predictors (from study):** Rainfall and humidity were identified as the most important variables for deficiency risk prediction.")
    
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
# Setup crop data from thresholds
# -----------------------------
crop_thresholds = thresholds.get("crop_thresholds", {})
crop_list = sorted(crop_thresholds.keys()) if crop_thresholds else []

# Get agronomic groups
agronomic_groups = thresholds.get("agronomic_groups", {
    "high_n_demand": ["rice", "maize", "cotton", "jute", "coffee"],
    "legumes": ["chickpea", "kidneybeans", "pigeonpeas", "mothbeans", "mungbean", "blackgram", "lentil"],
    "fruits": ["pomegranate", "banana", "mango", "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya", "coconut"]
})

# Reverse mapping: crop -> group
crop_to_group = {}
for group, crops in agronomic_groups.items():
    for crop in crops:
        crop_to_group[crop] = group

group_colors = {
    "high_n_demand": "🔵 High-N-Demand",
    "legumes": "🟢 Legumes",
    "fruits": "🟠 Fruits"
}

# -----------------------------
# UI
# -----------------------------
st.title("🌱 Predictive Analytics: Nutrient Deficiency Risk")
st.caption("Proactive risk estimation using environmental and soil-condition variables (temperature, humidity, pH, rainfall).")

tabs = st.tabs([
    "📊 Predict (Single Input)",
    "🔍 What-if Analysis",
    "📁 Batch Prediction",
    "📈 Model Performance",
    "📖 About / Target Definition"
])

# -----------------------------
# Tab 1: Single prediction WITH CROP SELECTION
# -----------------------------
with tabs[0]:
    st.subheader("Single Prediction")
    
    # Create two columns: left for crop selection, right for environmental inputs
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 🌾 Crop Selection")
        
        # Crop dropdown
        if crop_list:
            selected_crop = st.selectbox("Select Your Crop", crop_list)
            
            # Show crop group
            crop_group = crop_to_group.get(selected_crop, "unknown")
            st.markdown(f"**Crop Group:** {group_colors.get(crop_group, crop_group)}")
            
            # Show crop-specific thresholds
            if selected_crop in crop_thresholds:
                thr = crop_thresholds[selected_crop]
                st.info(f"""
                **📊 {selected_crop.upper()} - Deficiency Thresholds (25th percentile):**
                - Nitrogen (N) < {thr['N_thr']}
                - Phosphorus (P) < {thr['P_thr']}
                - Potassium (K) < {thr['K_thr']}
                
                > ⚠️ Risk is flagged if ANY nutrient falls below its threshold (Liebig's Law)
                """)
        else:
            selected_crop = None
            st.warning("No crop thresholds found. Please check thresholds.json file.")
    
    with col_right:
        st.markdown("### 🌡️ Environmental Conditions")
        temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0, step=0.1)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0, step=0.1)
        ph = st.number_input("Soil pH", 3.0, 10.0, 6.5, step=0.01)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, 100.0, step=0.1)
    
    # Predict button
    if st.button("🔮 Predict Risk", type="primary"):
        pred, prob = predict_one(model, features, temperature, humidity, ph, rainfall)
        band, color = risk_band(prob if prob is not None else 0.0)
        
        st.markdown("---")
        st.markdown("### 📋 Prediction Result")
        
        # Display results
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            if pred == 1:
                if selected_crop:
                    st.error(f"⚠️ **DEFICIENCY RISK DETECTED** for {selected_crop.upper()}")
                else:
                    st.error("⚠️ **DEFICIENCY RISK DETECTED**")
                st.metric("Predicted Class", "Deficiency Risk (1)")
            else:
                if selected_crop:
                    st.success(f"✅ **NORMAL CONDITION** for {selected_crop.upper()}")
                else:
                    st.success("✅ **NORMAL CONDITION**")
                st.metric("Predicted Class", "Normal (0)")
            st.metric("Risk Probability", f"{prob*100:.2f}%")
            st.markdown(f"**Risk Band:** :{color}[{band}]")
        
        with col_res2:
            if prob is not None:
                gauge_fig = make_risk_gauge(prob)
                st.plotly_chart(gauge_fig, use_container_width=True)
        
        st.markdown("### 📋 Recommendations")
        recs = recommendation_text(pred, prob, temperature, humidity, ph, rainfall, selected_crop, crop_thresholds)
        for r in recs:
            st.write(r)
        
        # Download report
        report = pd.DataFrame([{
            "crop": selected_crop if selected_crop else "unknown",
            "crop_group": crop_to_group.get(selected_crop, "unknown") if selected_crop else "unknown",
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
            "predicted_class": pred,
            "risk_probability": prob
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
    ### How the Deficiency Risk Label Was Constructed
    
    This prototype predicts a binary **Nutrient Deficiency Risk** label derived from soil Nitrogen (N), 
    Phosphorus (P), and Potassium (K) values — but **N, P, K are NOT used as model inputs** to avoid 
    circular prediction.
    
    **Target Rule (Liebig's Law of the Minimum):**
    > A record is labeled **Deficiency Risk (1)** if **ANY** of N, P, or K falls below its 
    > **crop-specific 25th percentile threshold** computed from the training set.
    
    This OR-based rule reflects the agronomic principle that crop growth is constrained by the 
    **most limiting macronutrient** (Brady & Weil, 2017; Marschner, 2012).
    
    ### Expected vs. Observed Flag Rates
    
    With three independent 25th-percentile thresholds, the theoretical probability of being flagged is:
    
    1 - (0.75)³ = 0.578 (57.8%)
                
    **Observed flag rates in this dataset:**
    - Fruits: 52.9%
    - High-N-Demand crops: 56.3%
    - Legumes: 54.0%
    
    The close match confirms the OR rule behaves as expected, with slight deviation due to weak positive 
    correlations among N, P, and K.
    
    ### Crop-Specific Thresholds (25th Percentile Examples)
    
    The 22 crops were organized into three agronomic demand classes:
    
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
    
    **Model inputs are ONLY these four variables.** N, P, K values are used only for label construction.
    
    ### Model Performance Summary
    
    - **Best Model:** Random Forest
    - **Test Accuracy:** 76.1%
    - **Recall (Deficiency Risk):** 85.8% — correctly flags ~86 of 100 actual deficiency cases
    - **ROC-AUC:** 0.854 — strong discriminative ability
    - **Top Predictors:** Rainfall (0.301), Humidity (0.291)
    
    > ⚠️ **Limitation:** This label is a distribution-derived proxy, not a field-validated diagnosis. 
    > Use as a screening tool to prioritize soil testing, not as a replacement for professional 
    > agronomic advice.
    """)
    
    with st.expander("📁 View Complete Crop-Specific Thresholds (JSON)"):
        st.json(thresholds)