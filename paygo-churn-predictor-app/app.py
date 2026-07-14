import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Page Configuration
# ===============================

st.set_page_config(
    page_title="PAYGo Solar Churn Dashboard",
    page_icon="☀️",
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



    # ---------------------------------------------
    # Select Risk Segment
    # ---------------------------------------------

    selected_risk = st.selectbox(
        "Select Risk Segment",
        [
            "All",
            "High Risk",
            "Medium Risk",
            "Low Risk"
        ]
    )

    # Filter the dataframe

    if selected_risk == "All":
        filtered_df = dashboard_df.copy()
    else:
        filtered_df = dashboard_df[
            dashboard_df["risk_category"] == selected_risk
        ]

    # ---------------------------------------------
    # Segment Summary Metrics
    # ---------------------------------------------

    st.subheader("Risk Segment Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Customers",
            f"{len(filtered_df):,}"
        )

    with col2:
        st.metric(
            "Average Churn Probability",
            f"{filtered_df['churn_probability'].mean():.2%}"
        )

    with col3:
        st.metric(
            "Average Repayment Rate",
            f"{filtered_df['repayment_rate'].mean():.1f}%"
        )

    st.divider()

    # ---------------------------------------------
    # Top Risk Customers
    # ---------------------------------------------

    st.subheader("Top Customers in Selected Segment")

    top_customers = (
        filtered_df
        .sort_values(
            "churn_probability",
            ascending=False
        )
        .head(10)
    )

    top_columns = [
        "customer_id",
        "risk_category",
        "churn_probability",
        "repayment_rate",
        "missed_payments"
    ]

    available_top_columns = [
        col for col in top_columns
        if col in filtered_df.columns
    ]

    st.dataframe(
        top_customers[available_top_columns],
        use_container_width=True
    )

    st.divider()

    # ---------------------------------------------
    # Customer Selection
    # ---------------------------------------------

    st.subheader("Select Customer")

    selected_customer = st.selectbox(
        "Customer ID",
        filtered_df["customer_id"]
    )

    customer = filtered_df[
        filtered_df["customer_id"] == selected_customer
    ].iloc[0]

    # ---------------------------------------------
    # Prediction Output
    # ---------------------------------------------

    st.subheader("Prediction Output")

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

    # ---------------------------------------------
    # Suggested Retention Action
    # ---------------------------------------------

    st.subheader("Suggested Retention Action")

    if customer["risk_category"] == "High Risk":

        st.error(
            """
            Immediate intervention recommended.

            • Prioritize customer follow-up.

            • Review repayment behaviour and missed payments.

            • Engage customer support if payment issues exist.

            • Consider targeted retention campaigns.
            """
        )

    elif customer["risk_category"] == "Medium Risk":

        st.warning(
            """
            Moderate intervention recommended.

            • Monitor payment behaviour.

            • Schedule follow-up engagement.

            • Encourage continued product usage.
            """
        )

    else:

        st.success(
            """
            Customer is currently low risk.

            • Maintain regular engagement.

            • Continue monitoring customer behaviour.
            """
        )

    st.divider()

    # ---------------------------------------------
    # Customer Profile
    # ---------------------------------------------

    st.subheader("Customer Profile")

    profile_columns = [

        "tenure",
        "age",
        "gender",
        "repayment_rate",
        "missed_payments",
        "outstanding_balance",
        "total_amount_paid",
        "payment_3m",
        "avg_payment_3m",
        "payment_trend_3m",
        "last_paid_amount",
        "last_paid_month",
        "total_calls",
        "outbound_calls",
        "received_calls",
        "unanswered_calls",
        "repossession_tickets",
        "complaint_tickets"

    ]

    available_columns = [

        col for col in profile_columns
        if col in dashboard_df.columns

    ]

    st.dataframe(
        customer[available_columns],
        use_container_width=True
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
