import streamlit as st
import pandas as pd

def decision_insights(df):
    st.subheader("📌 Decision-Level Insights")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns available for decision insights")
        return

    metric = st.selectbox("Select key metric", numeric_cols)

    mean_val = df[metric].mean()
    std_val = df[metric].std()

    st.write(f"Average {metric}: {mean_val:.2f}")
    st.write(
        "Stability:",
        "Volatile" if std_val > mean_val else "Stable"
    )
