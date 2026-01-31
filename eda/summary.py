import streamlit as st
import pandas as pd

def data_summary(df):
    st.subheader("📊 Dataset Summary")

    summary_data = []

    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)

        if unique_ratio < 0.05:
            inferred_type = "Categorical-like"
        elif unique_ratio > 0.9:
            inferred_type = "Identifier-like"
        else:
            inferred_type = "Continuous Numeric"

        summary_data.append({
            "Column": col,
            "Pandas Type": str(df[col].dtype),
            "Inferred Type": inferred_type,
            "Missing %": round(df[col].isnull().mean() * 100, 2),
            "Unique %": round(unique_ratio * 100, 2)
        })

    summary_df = pd.DataFrame(summary_data)

    st.dataframe(summary_df)

    st.markdown("---")
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(include="all"))
