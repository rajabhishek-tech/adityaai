import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_matrix(df):
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric columns for correlation")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)
