Nutrient Deficiency Risk Predictor

A machine learning web application that predicts the risk of soil nutrient deficiency based on environmental and soil conditions such as temperature, humidity, pH, and rainfall.

Built using Python, Machine Learning, and Streamlit.

Features
Predict nutrient deficiency risk (Low, Moderate, High)
Probability-based output
What-if analysis (sensitivity testing)
Batch prediction via CSV upload
Model performance visualization (Confusion Matrix, ROC Curve)
Automated recommendations based on input conditions

Model Overview
Target: Binary classification
0 → Normal
1 → Deficiency Risk
Based on:
Temperature
Humidity
Soil pH
Rainfall

Model is trained and stored in:

artifacts/best_model.pkl
Installation
1. Clone the repository
git clone https://github.com/taro-miz/nutrient-deficiency-predictor.git
cd nutrient-deficiency-predictor
2. Install dependencies
pip install -r requirements.txt

Dependencies used:
streamlit
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib

How to Run
streamlit run app.py

Then open the browser link shown in the terminal (usually http://localhost:8501).

Project Structure
.
├── app.py
├── requirements.txt
├── artifacts/
│   ├── best_model.pkl
│   ├── feature_list.pkl
│   ├── results_df.csv
│   ├── cv_df.csv
│   ├── confusion_matrix.npy
│   ├── roc_auc.txt
│   ├── thresholds.json
│   └── ...
Model Evaluation
Includes:
Confusion Matrix
ROC Curve
Cross-validation results
Notes
This is a prototype for predictive analytics.
The model does not directly use N, P, K values for prediction to avoid data leakage.
Predictions are based on environmental factors only.
Author
Miz
Future Improvements
Improve dataset size and diversity
Deploy to cloud (e.g., Streamlit Cloud)
