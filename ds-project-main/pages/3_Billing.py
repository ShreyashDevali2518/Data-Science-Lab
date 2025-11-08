import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

st.title("💰 ML-Based Billing Prediction System")

# Resolve dataset path relative to the project root
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models"
MODEL_PATH.mkdir(exist_ok=True)

# Load or train the model
@st.cache_resource
def load_or_train_model():
    model_file = MODEL_PATH / "billing_model.joblib"
    encoders_file = MODEL_PATH / "encoders.joblib"
    
    if model_file.exists() and encoders_file.exists():
        model = joblib.load(model_file)
        encoders = joblib.load(encoders_file)
        return model, encoders
    
    # If no saved model, train a new one
    df = pd.read_csv(ROOT / "healthcare_dataset_cleaned.csv")
    
    # Date features
    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
    df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])
    df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
    df["Admission Month"] = df["Date of Admission"].dt.month
    df["Admission Day"] = df["Date of Admission"].dt.day
    df["Admission Weekday"] = df["Date of Admission"].dt.weekday
    df["Discharge Month"] = df["Discharge Date"].dt.month
    df["Discharge Weekday"] = df["Discharge Date"].dt.weekday
    
    # Drop text columns
    drop_cols = [
        "Name", "Doctor", "Hospital", "Medication",
        "Test Results", "Date of Admission", "Discharge Date"
    ]
    df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Identify categorical columns
    cat_cols = df_model.select_dtypes(include=["object"]).columns.tolist()
    
    # Prepare encoders dictionary
    encoders = {
        "label_encode_cols": [],
        "le_map": {},
        "target_maps": {}
    }
    
    # Label encode low-cardinality columns
    label_encode_cols = [col for col in cat_cols if df_model[col].nunique() <= 10]
    target_encode_cols = [col for col in cat_cols if col not in label_encode_cols]
    
    for col in label_encode_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        encoders["le_map"][col] = le
    
    encoders["label_encode_cols"] = label_encode_cols
    
    # Target encode high-cardinality columns
    for col in target_encode_cols:
        means = df_model.groupby(col)["Billing Amount"].mean()
        df_model[col] = df_model[col].map(means)
        encoders["target_maps"][col] = means
    
    # Train model
    y = df_model["Billing Amount"]
    X = df_model.drop(columns=["Billing Amount"])
    
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.07,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X, y)
    
    # Save model and encoders
    joblib.dump(model, model_file)
    joblib.dump(encoders, encoders_file)
    
    return model, encoders

# Load/train model and get options from dataset
try:
    # Load the dataset first
    df = pd.read_csv(ROOT / "healthcare_dataset_cleaned.csv")
    
    # Then train/load model
    model, encoders = load_or_train_model()
    
    if not hasattr(model, 'get_booster'):
        st.error("Model not properly initialized. Please try refreshing the page.")
        st.stop()
        
    st.success("✅ ML model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Helper to safely get unique values
def unique_or_default(col_name, default_list):
    if col_name in df.columns:
        vals = df[col_name].dropna().unique().tolist()
        return sorted(vals) if vals else default_list
    return default_list

# Options for inputs
gender_options = unique_or_default("Gender", ["Male", "Female", "Other"])
blood_options = unique_or_default("Blood Type", ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"])
condition_options = unique_or_default("Medical Condition", 
    ["Diabetes", "Asthma", "Cancer", "Obesity", "Heart Disease", "Other"])
admission_options = unique_or_default("Admission Type", ["Emergency", "Urgent", "Elective"])
insurance_options = unique_or_default("Insurance Provider", 
    ["Private", "Government", "Blue Cross", "Aetna", "UnitedHealthcare", "None"])


def predict_bill(age, gender, blood_type, medical_condition,
                admission_type, insurance_provider,
                admission_date, discharge_date):
    """Predict bill using trained ML model"""
    ad = pd.to_datetime(admission_date)
    dd = pd.to_datetime(discharge_date)
    
    los = (dd - ad).days
    adm_month = ad.month
    adm_day = ad.day
    adm_weekday = ad.weekday()
    dis_month = dd.month
    dis_weekday = dd.weekday()
    
    # Get model's feature names
    feature_names = model.get_booster().feature_names
    
    # Prepare base features
    patient = {
        "Age": age,
        "Length of Stay": los,
        "Admission Month": adm_month,
        "Admission Day": adm_day,
        "Admission Weekday": adm_weekday,
        "Discharge Month": dis_month,
        "Discharge Weekday": dis_weekday,
    }
    
    # Process categorical columns
    categoricals = {
        "Gender": gender,
        "Blood Type": blood_type,
        "Admission Type": admission_type,
        "Medical Condition": medical_condition,
        "Insurance Provider": insurance_provider
    }
    
    for col, value in categoricals.items():
        if col in encoders["label_encode_cols"]:
            try:
                patient[col] = encoders["le_map"][col].transform([value])[0]
            except:
                patient[col] = 0  # fallback
        elif col in encoders["target_maps"]:
            patient[col] = encoders["target_maps"][col].get(
                value, encoders["target_maps"][col].mean()
            )
        else:
            patient[col] = 0
    
    # Create DataFrame with exact feature names from model
    new_df = pd.DataFrame([patient])
    if feature_names:
        new_df = new_df.reindex(columns=feature_names, fill_value=0)
    else:
        # Fallback to original columns if feature names not available
        X_cols = [c for c in df.columns if c not in ['Billing Amount'] + drop_cols]
        new_df = new_df.reindex(columns=X_cols, fill_value=0)
    
    return model.predict(new_df)[0]


# Drop text columns for prediction
drop_cols = [
    "Name", "Doctor", "Hospital", "Medication",
    "Test Results", "Date of Admission", "Discharge Date"
]

# Input form
st.subheader("Enter Patient Details for Bill Prediction")
with st.form(key="billing_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Gender", gender_options)
        blood_type = st.selectbox("Blood Type", blood_options)
        medical_condition = st.selectbox("Medical Condition", condition_options)
    
    with col2:
        admission_type = st.selectbox("Admission Type", admission_options)
        insurance_provider = st.selectbox("Insurance Provider", insurance_options)
        admission_date = st.date_input("Admission Date", 
            value=pd.to_datetime('today').date())
        discharge_date = st.date_input("Discharge Date", 
            value=pd.to_datetime('today').date())
    
    submitted = st.form_submit_button("Predict Bill")

if submitted:
    # Basic validation
    if discharge_date < admission_date:
        st.error("❌ Discharge date cannot be earlier than admission date!")
    else:
        # Get prediction
        estimated_bill = predict_bill(
            age, gender, blood_type, medical_condition,
            admission_type, insurance_provider,
            admission_date, discharge_date
        )

        # Show results
        st.write("### 🧾 Bill Prediction Results")
        st.write(f"**Estimated bill:** $ {estimated_bill:,.2f}")

        # Show factors that influenced the prediction
        st.write("\n### 📊 Key Factors")
        st.write("- **Length of stay:** {} days".format(
            (pd.to_datetime(discharge_date) - pd.to_datetime(admission_date)).days
        ))

        # Compare to similar cases
        similar_condition = df[df["Medical Condition"] == medical_condition]
        if len(similar_condition) > 0:
            avg_similar = similar_condition["Billing Amount"].mean()
            st.write(f"- **Average bill for {medical_condition}:** $ {avg_similar:,.2f}")

        similar_admission = df[df["Admission Type"] == admission_type]
        if len(similar_admission) > 0:
            avg_admission = similar_admission["Billing Amount"].mean()
            st.write(f"- **Average bill for {admission_type} admissions:** $ {avg_admission:,.2f}")

        # Add explanation
        st.info("""
        💡 **How this works:** This prediction uses machine learning trained on historical billing data.
        It considers factors like length of stay, medical condition, admission type, and more to estimate the bill.
        The model is regularly updated with new data to improve accuracy.
        """)

        st.info("Note: This is an estimate based on a trained XGBoost model. For production use, validate with additional data and domain expertise.")

# Show dataset statistics
if st.checkbox("Show Dataset Statistics"):
    st.subheader("📈 Billing Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Average bills by condition:**")
        condition_stats = df.groupby("Medical Condition")["Billing Amount"].agg(
            ["mean", "count"]
        ).sort_values("mean", ascending=False)

        condition_stats.columns = ["Average Bill ($)", "Number of Cases"]
        st.write(condition_stats.style.format({
            "Average Bill ($)": "${:,.2f}"
        }).to_html(), unsafe_allow_html=True)

    with col2:
        st.write("**Average bills by admission type:**")
        admission_stats = df.groupby("Admission Type")["Billing Amount"].agg(
            ["mean", "count"]
        ).sort_values("mean", ascending=False)

        admission_stats.columns = ["Average Bill ($)", "Number of Cases"]
        st.write(admission_stats.style.format({
            "Average Bill ($)": "${:,.2f}"
        }).to_html(), unsafe_allow_html=True)


