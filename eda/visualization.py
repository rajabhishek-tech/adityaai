import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def recommend_chart(series):
    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_ratio = numeric_series.notna().mean()

    if numeric_ratio > 0.8:
        skew = numeric_series.dropna().skew()
        if abs(skew) > 1:
            return "Box Plot"
        return "Histogram"

    unique_ratio = series.nunique() / len(series)
    if unique_ratio < 0.2:
        return "Bar Chart"
    return "Pie Chart"



def visualize_data(df):
    st.subheader("📈 Data Visualizations")

    col = st.selectbox("Select column", df.columns)
    series = df[col]

    # Try numeric conversion
    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_ratio = numeric_series.notna().mean()

    # Try datetime conversion
    datetime_series = pd.to_datetime(series, errors="coerce")
    datetime_ratio = datetime_series.notna().mean()

    # ---------------- NUMERIC ----------------
    if numeric_ratio > 0.8:
        st.success("Detected Numeric Column")

        chart_type = st.selectbox(
            "Choose chart type",
            [
                "Histogram",
                "Box Plot",
                "Violin Plot",
                "KDE Plot",
                "Line Plot (Trend)",
                "Scatter Plot",
                "3D Scatter (Advanced)"
            ]
        )

        data = numeric_series.dropna()

        # Matplotlib / Seaborn charts
        if chart_type in ["Histogram", "Box Plot", "Violin Plot", "KDE Plot", "Line Plot (Trend)", "Scatter Plot"]:
            fig, ax = plt.subplots()

            if chart_type == "Histogram":
                sns.histplot(data, kde=True, ax=ax)

            elif chart_type == "Box Plot":
                sns.boxplot(x=data, ax=ax)

            elif chart_type == "Violin Plot":
                sns.violinplot(x=data, ax=ax)

            elif chart_type == "KDE Plot":
                sns.kdeplot(data, fill=True, ax=ax)

            elif chart_type == "Line Plot (Trend)":
                ax.plot(data.values)

            elif chart_type == "Scatter Plot":
                ax.scatter(range(len(data)), data)

            ax.set_title(f"{chart_type} for {col}")
            st.pyplot(fig)

        # 3D Scatter (Plotly)
        elif chart_type == "3D Scatter (Advanced)":
            df_3d = pd.DataFrame({
                "Index": range(len(data)),
                col: data
            })

            fig = px.scatter_3d(
                df_3d,
                x="Index",
                y=col,
                z=df_3d.index,
                title=f"3D Scatter Plot for {col}"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- DATETIME ----------------
    elif datetime_ratio > 0.8:
        st.info("Detected Datetime Column")

        chart_type = st.selectbox(
            "Choose chart type",
            ["Line Plot", "Area Plot"]
        )

        clean_dates = datetime_series.dropna().sort_values()

        fig, ax = plt.subplots()

        if chart_type == "Line Plot":
            ax.plot(clean_dates, range(len(clean_dates)))

        elif chart_type == "Area Plot":
            ax.fill_between(range(len(clean_dates)), clean_dates.index)

        ax.set_title(f"{chart_type} for {col}")
        st.pyplot(fig)

    # ---------------- CATEGORICAL ----------------
    else:
        st.info("Detected Categorical Column")

        chart_type = st.selectbox(
            "Choose chart type",
            ["Bar Chart", "Count Plot", "Pie Chart", "Donut Chart"]
        )

        counts = series.astype(str).value_counts().head(10)

        # Bar / Count
        if chart_type in ["Bar Chart", "Count Plot"]:
            fig, ax = plt.subplots()
            sns.barplot(x=counts.values, y=counts.index, ax=ax)
            ax.set_title(f"{chart_type} for {col}")
            st.pyplot(fig)

        # Pie
        elif chart_type == "Pie Chart":
            fig, ax = plt.subplots()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
            ax.set_title(f"Pie Chart for {col}")
            st.pyplot(fig)

        # Donut
        elif chart_type == "Donut Chart":
            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                wedgeprops=dict(width=0.4)
            )
            ax.set_title(f"Donut Chart for {col}")
            st.pyplot(fig)
