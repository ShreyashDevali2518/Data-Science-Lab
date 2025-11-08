import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("📊 Exploratory Data Analysis")

df = pd.read_csv("healthcare_dataset_cleaned.csv")

st.subheader("📈 Basic Statistics")
# Use st.dataframe for better readability and avoid color issues
st.dataframe(df.describe(include='all'), use_container_width=True)

st.subheader("🩺 Distribution of Age")
fig, ax = plt.subplots()
sns.histplot(df["Age"], kde=True, bins=20, ax=ax, color='skyblue')
st.pyplot(fig)

st.subheader("💉 Medical Condition Count")
fig, ax = plt.subplots()
sns.countplot(y="Medical Condition", data=df, ax=ax, order=df["Medical Condition"].value_counts().index, palette='viridis')
st.pyplot(fig)

st.subheader("🩸 Blood Type Distribution")
fig, ax = plt.subplots()
sns.countplot(x="Blood Type", data=df, ax=ax, order=df["Blood Type"].value_counts().index, palette='plasma')
st.pyplot(fig)

st.subheader("📅 Admission Type Distribution")
fig, ax = plt.subplots()
sns.countplot(x="Admission Type", data=df, ax=ax, order=df["Admission Type"].value_counts().index, palette='coolwarm')
st.pyplot(fig)

st.subheader("🔥 Correlation Heatmap (Numerical Features Only)")
numeric_df = df.select_dtypes(include=['int64', 'float64'])  # Only numeric columns
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
