import streamlit as st
import pandas as pd
import numpy as np

def generate_ai_insights(df):
    st.subheader("🤖 AI-Generated Column Insights")

    col = st.selectbox("Select column for insights", df.columns)
    series = df[col]

    if pd.api.types.is_numeric_dtype(series):
        st.write(f"• Mean: {series.mean():.2f}")
        st.write(f"• Min: {series.min()}")
        st.write(f"• Max: {series.max()}")

        if series.skew() > 1:
            st.write("• Right-skewed distribution (possible outliers)")
        elif series.skew() < -1:
            st.write("• Left-skewed distribution")

    elif pd.api.types.is_object_dtype(series):
        top = series.value_counts().idxmax()
        st.write(f"• Most frequent value: {top}")
        st.write(f"• Unique values: {series.nunique()}")

    elif pd.api.types.is_datetime64_any_dtype(series):
        st.write("• Datetime column detected")
        st.write(f"• Range: {series.min()} → {series.max()}")

    else:
        st.write("• Unsupported column type")
