import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================
# PAGE TITLE
# ==============================================
st.title("📊 Exploratory Data Analysis (EDA)")

# ==============================================
# LOAD DATA
# ==============================================
try:
    df = pd.read_csv("healthcare_dataset_cleaned.csv")
    st.success("✅ Dataset loaded successfully!")
    st.dataframe(df.head())
except Exception as e:
    st.error("❌ Could not load dataset. Please check if 'healthcare_dataset_cleaned.csv' exists.")
    st.stop()

# ==============================================
# DATE CONVERSION + NEW FEATURE
# ==============================================
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")

# Create Length of Stay
df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days

st.subheader("🕒 Length of Stay (Days)")
st.write(df[["Date of Admission", "Discharge Date", "Length of Stay"]].head())

# ==============================================
# SCATTER PLOT: Length of Stay vs Billing Amount
# ==============================================
st.subheader("📈 Length of Stay vs Billing Amount")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x="Length of Stay", y="Billing Amount", color="royalblue", ax=ax)
plt.title("Length of Stay vs Billing Amount")
st.pyplot(fig)

# ==============================================
# BOX PLOT: Billing by Medical Condition
# ==============================================
st.subheader("💊 Billing Amount by Medical Condition")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x="Medical Condition", y="Billing Amount", palette="viridis", ax=ax)
plt.xticks(rotation=45)
plt.title("Billing Amount by Medical Condition")
st.pyplot(fig)

# ==============================================
# BOX PLOT: Billing by Admission Type
# ==============================================
st.subheader("🏥 Billing Amount by Admission Type")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x="Admission Type", y="Billing Amount", palette="coolwarm", ax=ax)
plt.xticks(rotation=45)
plt.title("Billing Amount by Admission Type")
st.pyplot(fig)

# ==============================================
# CORRELATION HEATMAP
# ==============================================
st.subheader("🔥 Correlation Heatmap (Numeric Features)")
numeric_df = df.select_dtypes(include=["float64", "int64"])
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
plt.title("Correlation Heatmap")
st.pyplot(fig)

st.success("✅ EDA Completed Successfully!")
