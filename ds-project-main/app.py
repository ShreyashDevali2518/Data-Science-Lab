import streamlit as st
from pathlib import Path

# Set page config for the entire app
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Main page content
st.title("🏥 Healthcare Analytics Dashboard")

st.markdown("""
### Welcome to the Healthcare Analytics System

This application provides tools for:

1. **🧹 Data Cleaning** (Page 1)
   - View and verify cleaned healthcare dataset
   - Check for missing values
   - Data quality assessment

2. **📊 Exploratory Data Analysis** (Page 2)
   - Visualize patient demographics
   - Analyze medical conditions
   - Understand admission patterns

3. **💰 Billing Estimation** (Page 3)
   - Estimate patient bills based on multiple factors
   - View cost breakdowns
   - Insurance coverage analysis

### Getting Started

Choose a page from the sidebar to:
- Review the cleaned dataset
- Explore healthcare insights
- Calculate estimated bills

### Dataset Overview
The analysis is based on our healthcare dataset which includes:
- Patient demographics
- Medical conditions
- Admission details
- Insurance information
- Length of stay
- Billing information
""")

# Quick dataset stats if available
try:
    ROOT = Path(__file__).resolve().parent
    df = pd.read_csv(ROOT / "healthcare_dataset_cleaned.csv")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Patients", len(df))

    with col2:
        if "Medical Condition" in df.columns:
            n_conditions = df["Medical Condition"].nunique()
            st.metric("Unique Conditions", n_conditions)

    with col3:
        if "Admission Type" in df.columns:
            emergency_count = len(df[df["Admission Type"] == "Emergency"])
            st.metric("Emergency Cases", emergency_count)

except Exception as e:
    st.info("Load the dataset through the Data Cleaning page to see statistics.")
