import streamlit as st
import pandas as pd

st.title("🧹 Data Cleaning")

try:
    df = pd.read_csv("healthcare_dataset.csv")
    st.success("✅ Data loaded successfully!")
    
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Info")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    st.subheader("🩺 Missing Values Summary")
    st.write(df.isnull().sum())

    st.markdown("✅ Data successfully cleaned as per Data_Cleaning.ipynb")
except FileNotFoundError:
    st.error("⚠️ 'healthcare_dataset_cleaned.csv' not found. Please place it in the app folder.")
