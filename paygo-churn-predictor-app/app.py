import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Page Configuration
# ===============================

st.set_page_config(
    page_title="PAYGo Solar Churn Dashboard",
    page_icon="*",
    layout="wide"
)


# ===============================
# Load Data
# ===============================

@st.cache_data
def load_data():
    dashboard_df = pd.read_csv(
        "customer_churn_dashboard_data.csv"
    )

    shap_df = pd.read_csv(
        "shap_feature_importance.csv"
    )

    model_df = pd.read_csv(
        "model_performance.csv"
    )

    return dashboard_df, shap_df, model_df


dashboard_df, shap_df, model_df = load_data()

# ===============================
# Title
# ===============================

st.title(
    "PAYGo Solar Customer Churn Prediction Dashboard"
)

st.markdown(
    """
    This dashboard presents machine learning-based customer churn predictions,
    risk segmentation, and explainable AI insights to support customer retention decisions.
    """
)

# ===============================
# Sidebar Navigation
# ===============================

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview Dashboard",
        "Customer Risk Explorer",
        "Churn Drivers (SHAP)",
        "Model Performance"
    ]
)

# ==========================================================
# PAGE 1: OVERVIEW DASHBOARD
# ==========================================================

if page == "Overview Dashboard":

    st.header("Customer Churn Overview")

    total_customers = len(dashboard_df)

    churn_rate = (
            dashboard_df["churn"].mean() * 100
    )

    high_risk = (
        dashboard_df["risk_category"]
        .eq("High Risk")
        .sum()
    )

    medium_risk = (
        dashboard_df["risk_category"]
        .eq("Medium Risk")
        .sum()
    )

    low_risk = (
        dashboard_df["risk_category"]
        .eq("Low Risk")
        .sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

    with col3:
        st.metric(
            "High Risk Customers",
            f"{high_risk:,}"
        )

    with col4:
        st.metric(
            "Medium Risk Customers",
            f"{medium_risk:,}"
        )

    st.divider()

    # Risk segmentation

    st.subheader(
        "Customer Risk Segmentation"
    )

    risk_counts = (
        dashboard_df["risk_category"]
        .value_counts()
    )

    st.bar_chart(
        risk_counts
    )

    st.subheader(
        "Churn Distribution"
    )

    churn_counts = (
        dashboard_df["churn"]
        .value_counts()
        .rename(
            {
                0: "Active Customers",
                1: "Churned Customers"
            }
        )
    )

    st.bar_chart(
        churn_counts
    )





# ==========================================================
# PAGE 2: CUSTOMER RISK EXPLORER
# ==========================================================

elif page == "Customer Risk Explorer":

    st.header(
        "Customer Risk Explorer"
    )

    selected_customer = st.selectbox(
        "Select Customer ID",
        dashboard_df["customer_id"]
    )

    customer = dashboard_df[
        dashboard_df["customer_id"]
        == selected_customer
        ].iloc[0]

    st.subheader(
        "Prediction Output"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Churn Probability",
            f"{customer['churn_probability']:.2%}"
        )

    with col2:

        st.metric(
            "Risk Category",
            customer["risk_category"]
        )

    st.divider()

    st.subheader(
        "Customer Profile"
    )

    profile_columns = [
        "tenure",
        "age",
        "gender",
        "repayment_rate",
        "missed_payments",
        "outstanding_balance",
        "total_amount_paid",
        "total_calls"
    ]

    available_columns = [
        col for col in profile_columns
        if col in dashboard_df.columns
    ]

    st.dataframe(
        customer[available_columns]
    )





# ==========================================================
# PAGE 3: SHAP EXPLAINABILITY
# ==========================================================

elif page == "Churn Drivers (SHAP)":

    st.header(
        "Explainable AI - Churn Drivers"
    )

    st.write(
        """
        SHAP values explain the contribution of each feature
        to customer churn predictions.
        """
    )

    top_features = (
        shap_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(10)
    )

    st.subheader(
        "Top Factors Influencing Churn"
    )

    st.bar_chart(
        top_features.set_index(
            "Feature"
        )
    )

    st.subheader(
        "Feature Importance Table"
    )

    st.dataframe(
        top_features
    )





# ==========================================================
# PAGE 4: MODEL PERFORMANCE
# ==========================================================

elif page == "Model Performance":

    st.header(
        "Model Evaluation"
    )

    st.write(
        """
        Comparison of machine learning models evaluated
        using Accuracy, Precision, Recall, F1-score and ROC-AUC.
        """
    )

    st.dataframe(
        model_df
    )

    st.subheader(
        "ROC-AUC Comparison"
    )

    st.bar_chart(
        model_df.set_index(
            "Model"
        )["ROC-AUC"]
    )

    best_model = model_df.loc[
        model_df["ROC-AUC"].idxmax()
    ]

    st.success(
        f"""
        Best Performing Model:
        {best_model['Model']}

        ROC-AUC:
        {best_model['ROC-AUC']:.3f}
        """
    )
