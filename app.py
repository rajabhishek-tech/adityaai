import streamlit as st
import pandas as pd
import base64
import time


st.set_page_config(page_title="AdityaAI – EDA Studio", layout="wide")
#add_background_image("assets/background.jpg")

# ---------------- APP WARM-UP ----------------
with st.spinner("🔄 Warming up AdityaAI engine..."):
    time.sleep(1.5) 


# ---------------- IMPORTS ----------------
from utils.loader import load_data

from eda.summary import data_summary
from eda.missing import missing_values
from eda.visualization import visualize_data
from eda.correlation import correlation_matrix

from insights.ai_insights import generate_ai_insights
from insights.column_health import analyze_column_health
from insights.dataset_health import calculate_dataset_health
from insights.domain_kpis import compute_domain_kpis
from insights.next_actions import generate_next_actions
from insights.ml_recommender import recommend_ml_strategy

from automl.preprocess import prepare_data
from automl.trainer import train_models
from automl.evaluator import evaluate_models
from automl.leaderboard import build_leaderboard
from automl.feature_importance import get_feature_importance

from automl.metrics import (
    plot_confusion_matrix,
    plot_regression_residuals,
    regression_metrics,
    plot_clusters_2d
)

from automl.clustering import (
    run_kmeans,
    run_dbscan,
    reduce_dimensions
)

from reports.pdf_report import generate_pdf


# ---------------- SESSION STATE ----------------
if "df" not in st.session_state:
    st.session_state.df = None


# ---------------- BACKGROUND ----------------
def add_background_image(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
        }}
        .block-container {{
            background-color: rgba(255,255,255,0.9);
            padding: 2rem;
            border-radius: 16px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


#st.set_page_config(page_title="AdityaAI – EDA Studio", layout="wide")

add_background_image("assets/background.jpg")



# ---------------- APP WARM-UP ----------------
#with st.spinner("🔄 Warming up AdityaAI engine..."):
   # time.sleep(1.5) 


# =========================
# 🏠 HOME
# =========================
def show_home():
    st.title("🚀 AdityaAI – Intelligent EDA Studio")

    uploaded = st.file_uploader("Upload CSV or Excel", ["csv", "xlsx"])
    if uploaded:
        df = load_data(uploaded)
        st.session_state.df = df
        st.success("Dataset loaded ✅")

        c1, c2 = st.columns(2)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])

        st.dataframe(df.head())


# =========================
# 📊 EDA
# =========================
def show_eda():
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first")
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Summary", "Missing", "Visuals", "Correlation"]
    )

    with tab1:
        data_summary(df)

    with tab2:
        missing_values(df)

    with tab3:
        visualize_data(df)

    with tab4:
        correlation_matrix(df)


# =========================
# 🤖 AI INSIGHTS
# =========================
def show_ai_insights():
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first")
        return

    st.title("🤖 AI Insights")

    # ---------- AI INSIGHTS ----------
    generate_ai_insights(df)

    # ---------- DATA HEALTH ----------
    health_df = analyze_column_health(df)
    health_score = calculate_dataset_health(health_df)
    st.metric("Dataset Health Score", f"{health_score}/100")

    st.markdown("---")

    # ===============================
    # 🔀 ML MODE SELECTOR
    # ===============================
    ml_mode = st.radio(
        "Select ML Mode",
        [
            "🧠 Supervised Learning (Prediction)",
            "🧩 Unsupervised Learning (Clustering)"
        ]
    )

    # =====================================================
    # 🧠 SUPERVISED AutoML
    # =====================================================
    if ml_mode.startswith("🧠"):

        st.subheader("🎯 Supervised AutoML")

        auto_plan = recommend_ml_strategy(df)

        target = st.selectbox(
            "Select Target Column",
            df.columns,
            index=df.columns.get_loc(auto_plan["suggested_target"])
            if auto_plan["suggested_target"] in df.columns else 0
        )

        y = df[target]

        if y.isna().any():
            st.error("Target contains missing values")
            return

        task_type = auto_plan["task_type"]
        st.info(f"Detected task type: **{task_type}**")

        if st.button("🚀 Run AutoML (Supervised)"):

            with st.spinner("Training models..."):
                preprocessor, X_train, X_test, y_train, y_test = prepare_data(
                    df, target, task_type
                )

                trained_models = train_models(
                    preprocessor, X_train, y_train, task_type
                )

                results = evaluate_models(
                    trained_models, X_test, y_test, task_type
                )

                leaderboard = build_leaderboard(results)

            st.subheader("🏆 Model Leaderboard")
            st.dataframe(leaderboard, use_container_width=True)

            best = leaderboard.iloc[0]
            best_model = trained_models[best["Model"]]

            # ---------- DIAGNOSTICS ----------
            st.subheader("📊 Model Diagnostics")

            if task_type != "Regression":
                fig = plot_confusion_matrix(best_model, X_test, y_test)
                st.pyplot(fig)
            else:
                fig = plot_regression_residuals(best_model, X_test, y_test)
                st.pyplot(fig)

                metrics = regression_metrics(y_test, best_model.predict(X_test))
                c1, c2 = st.columns(2)
                c1.metric("RMSE", metrics["RMSE"])
                c2.metric("R² Score", metrics["R2 Score"])

            # ---------- FEATURE IMPORTANCE ----------
            fi_df = get_feature_importance(best_model, preprocessor)
            if fi_df is not None:
                st.subheader("📊 Feature Importance")
                st.bar_chart(fi_df.set_index("Feature")["Importance"])
                st.dataframe(fi_df)

    # =====================================================
    # 🧩 CLUSTERING AutoML
    # =====================================================
    else:
        st.subheader("🧩 Clustering AutoML")

        numeric_df = df.select_dtypes(include=["int64", "float64"]).dropna()

        if numeric_df.shape[1] < 2:
            st.warning("Need at least 2 numeric columns for clustering")
            return

        if st.button("🚀 Run Clustering AutoML"):

            with st.spinner("Running clustering models..."):
                kmeans = run_kmeans(numeric_df)
                dbscan = run_dbscan(numeric_df)

                all_models = kmeans + dbscan

            cluster_df = pd.DataFrame([
                {
                    "Model": m["Model"],
                    "Clusters": m["Clusters"],
                    "Silhouette": m["Silhouette Score"]
                }
                for m in all_models
            ]).sort_values("Silhouette", ascending=False)

            st.subheader("🏆 Clustering Leaderboard")
            st.dataframe(cluster_df, use_container_width=True)

            best_cluster = all_models[cluster_df.index[0]]

            X_2d = reduce_dimensions(numeric_df)
            fig = plot_clusters_2d(
                X_2d,
                best_cluster["Labels"],
                "Best Clustering Result"
            )
            st.pyplot(fig)


# =========================
# 💼 BUSINESS INSIGHTS
# =========================
def show_business_insights():
    st.title("💼 Business Insights")

    df = st.session_state.df
    if df is None:
        return

    domain = st.selectbox(
        "Domain",
        ["General", "Retail", "Finance", "Marketing"]
    )

    kpis = compute_domain_kpis(df, domain)
    for k, v in kpis.items():
        st.metric(k, v)


# =========================
# 🧭 NAVIGATION
# =========================
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 EDA", "🤖 AI Insights", "💼 Business Insights"]
)

if page == "🏠 Home":
    show_home()
elif page == "📊 EDA":
    show_eda()
elif page == "🤖 AI Insights":
    show_ai_insights()
elif page == "💼 Business Insights":
    show_business_insights()


# ---------------- FEEDBACK ----------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 Give Feedback")

st.sidebar.markdown(
    """
    We’re actively improving **AdityaAI** 🚀  
    Share your feedback, feature requests, or bugs.
    
    👉 [Submit Feedback](https://forms.gle/https://docs.google.com/forms/d/e/1FAIpQLSdJtM2cM3G_P9ZynAEbF85GdK8u7bkPxYIDhZqhLvEV2syYJQ/viewform?usp=sharing&ouid=111831618243201512850)
    """,
    unsafe_allow_html=True
)

