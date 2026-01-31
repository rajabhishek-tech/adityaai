import streamlit as st
import pandas as pd

def missing_values(df):
    missing_df = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().mean() * 100).round(2)
    })

    missing_df = missing_df[missing_df["Missing Count"] > 0]

    if missing_df.empty:
        st.success("No missing values found 🎉")
    else:
        st.dataframe(missing_df)
