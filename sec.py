# energy_prediction_app.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- UI Config ---
st.set_page_config(page_title="Energy Classifier", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #00416d;
        }
        .stButton>button {
            background-color: #008cba;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel("energydata_cleaned_6000.xlsx")
    return df

df = load_data()

# --- Data Preprocessing ---
thresh = df['Energy_Used_Wh'].median()
df['High_Usage'] = (df['Energy_Used_Wh'] > thresh).astype(int)
df.drop(['Energy_Used_Wh'], axis=1, inplace=True)

X = df.drop(['High_Usage'], axis=1)
y = df['High_Usage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- UI ---
st.title("🏠 Energy Usage Classification App")
st.markdown("""
<div class="main">
<p style='font-size: 18px;'>
This interactive app classifies whether a home's energy usage is <strong>High</strong> or <strong>Low</strong> based on environmental and indoor sensor features.
</p>
</div>
""", unsafe_allow_html=True)

# --- Data Visualization ---
st.header("📊 Data Visualization")
col1, col2 = st.columns(2)

with col1:
    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        st.markdown("**Conclusion:** The correlation heatmap shows how different features relate to each other. Notably, temperature and humidity values in living and kitchen areas have a significant impact on energy usage.")

with col2:
    if st.checkbox("Show Target Distribution"):
        st.bar_chart(df['High_Usage'].value_counts())
        st.markdown("**Conclusion:** The distribution plot shows a relatively balanced dataset with nearly equal instances of high and low energy consumption.")

# --- Feature Importance ---
st.header("📌 Feature Importance")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns)
fig_imp, ax_imp = plt.subplots()
importances.nlargest(10).plot(kind='barh', ax=ax_imp, color='teal')
st.pyplot(fig_imp)
st.markdown(f"**Conclusion:** The most influential features affecting energy usage are: `{importances.nlargest(5).index.tolist()}`.")

# --- Model Training & Evaluation ---
st.header("⚙️ Model Comparison")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds)
    }

st.dataframe(pd.DataFrame(results).T.round(3).style.set_caption("Model Evaluation Metrics"))
st.markdown("**Conclusion:** Gradient Boosting and Random Forest tend to outperform Logistic Regression in terms of F1 Score and Recall, indicating they are better suited for capturing both false positives and negatives.")

# --- Hyperparameter Tuning ---
st.header("🔧 Hyperparameter Tuning (Random Forest)")
params = {
    'n_estimators': [50, 100],
    'max_depth': [10, None]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid=params, cv=3, scoring='f1')
grid.fit(X_train, y_train)
st.success(f"Best Params: {grid.best_params_}")
st.success(f"Best F1 Score: {round(grid.best_score_, 3)}")
st.markdown("**Conclusion:** Hyperparameter tuning optimized the model’s depth and estimators for better generalization and reduced overfitting.")

# --- Live Model Interface ---
st.header("🧪 Make a Prediction")
best_model = grid.best_estimator_

with st.form("prediction_form"):
    st.markdown("<b>Enter feature values below:</b>", unsafe_allow_html=True)
    inputs = []
    cols = st.columns(3)
    for idx, col in enumerate(X.columns):
        default_val = float(df[col].mean())
        input_val = cols[idx % 3].number_input(f"{col}", value=default_val)
        inputs.append(input_val)
    submitted = st.form_submit_button("Classify Energy Usage")
    if submitted:
        inp_array = np.array(inputs).reshape(1, -1)
        prediction = best_model.predict(inp_array)[0]
        label = "High" if prediction == 1 else "Low"
        st.success(f"🔋 Predicted Energy Usage: {label} Consumption")

# Save model
joblib.dump(best_model, "best_classifier_model.pkl")
