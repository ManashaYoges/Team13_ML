
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Team 13 | ML Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "classification_best_partA.pkl"
RESULTS_PATH = BASE_DIR / "classification_partA_results.csv"


# --------------------------------------------------
# 2. LOAD THE EXISTING TRAINED MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 3. SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("🧠 Team 13")
st.sidebar.caption("Machine Learning Studio")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Churn Prediction",
        "Model Comparison",
        "About Project"
    ]
)


# --------------------------------------------------
# 4. HOME PAGE
# --------------------------------------------------
if page == "Home":

    st.title("Machine Learning Prediction Dashboard")
    st.write(
        "An interactive dashboard for exploring our "
        "machine learning models and their predictions."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("ML Tracks", "3")
    col2.metric("Prediction Task", "Classification")
    col3.metric("Interface", "Streamlit")

    st.subheader("Explore the application")

    a, b = st.columns(2)

    with a:
        st.markdown("### 👥 Customer Churn")
        st.write(
            "Enter customer details and use the trained "
            "model to estimate the likelihood of churn."
        )

    with b:
        st.markdown("### 📈 Model Comparison")
        st.write(
            "Explore the classification algorithms and "
            "their saved evaluation metrics."
        )

    st.info(
        "Choose a page from the sidebar to get started."
    )


# --------------------------------------------------
# 5. CUSTOMER CHURN PREDICTION
# --------------------------------------------------
elif page == "Churn Prediction":

    st.title("Customer Churn Prediction")
    st.write(
        "Enter the customer's details below and click "
        "**Predict Churn**."
    )

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Could not load the saved model: {e}")
        st.stop()

    with st.form("customer_form"):

        st.subheader("Customer Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox(
                "Gender", ["Female", "Male"]
            )
            senior = st.selectbox(
                "Senior Citizen", [0, 1],
                format_func=lambda x: (
                    "Yes" if x == 1 else "No"
                )
            )
            partner = st.selectbox(
                "Has Partner", ["Yes", "No"]
            )
            dependents = st.selectbox(
                "Has Dependents", ["Yes", "No"]
            )
            tenure = st.slider(
                "Tenure (months)", 0, 72, 12
            )

        with c2:
            phone = st.selectbox(
                "Phone Service", ["Yes", "No"]
            )
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )
            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )

        with c3:
            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=1000.0,
                value=70.0,
                step=1.0
            )
            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=100000.0,
                value=840.0,
                step=10.0
            )
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        st.subheader("Services")

        s1, s2, s3 = st.columns(3)

        with s1:
            security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )
            backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )
            device = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

        with s2:
            tech = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )
            tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )
            movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        predict_button = st.form_submit_button(
            "🔍 Predict Churn",
            use_container_width=True
        )

    if predict_button:

        # Match the feature engineering used in the notebook.
        service_values = [
            phone, multiple_lines, internet, security,
            backup, device, tech, tv, movies
        ]

        inactive_values = [
            "No",
            "No internet service",
            "No phone service"
        ]

        num_services = sum(
            value not in inactive_values
            for value in service_values
        )

        # Use the original column names expected by
        # the saved preprocessing/model pipeline.
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "NumServices": num_services
        }])

        try:
            prediction = int(model.predict(customer)[0])

            st.divider()
            st.subheader("Prediction Result")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(customer)[0]
                classes = list(model.classes_)

                churn_index = classes.index(1)
                churn_probability = float(
                    probabilities[churn_index]
                )

                r1, r2 = st.columns(2)

                r1.metric(
                    "Predicted Churn Probability",
                    f"{churn_probability:.1%}"
                )

                r2.metric(
                    "Predicted Class",
                    "Churn" if prediction == 1 else "Retained"
                )

                st.progress(churn_probability)

                if prediction == 1:
                    st.warning(
                        "The model predicts that this "
                        "customer may churn."
                    )
                else:
                    st.success(
                        "The model predicts that this "
                        "customer will be retained."
                    )

                st.caption(
                    "This is a model estimate, not a "
                    "guarantee of future customer behaviour."
                )

            else:
                st.metric(
                    "Predicted Class",
                    "Churn" if prediction == 1 else "Retained"
                )
                st.info(
                    "This model does not provide "
                    "predict_proba probabilities."
                )

        except Exception as e:
            st.error(
                "Prediction failed. Check that the saved "
                f"model and input features match. Details: {e}"
            )


# --------------------------------------------------
# 6. MODEL COMPARISON PAGE
# --------------------------------------------------
elif page == "Model Comparison":

    st.title("Classification Model Comparison")
    st.write(
        "Explore evaluation results saved by the project."
    )

    if not RESULTS_PATH.exists():
        st.warning(
            "The classification results CSV was not found "
            "in the models folder."
        )
        st.stop()

    try:
        results = pd.read_csv(RESULTS_PATH)
    except Exception as e:
        st.error(f"Could not read results: {e}")
        st.stop()

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    # Adapt to the column names used in the CSV.
    columns = {
        str(col).strip().lower(): col
        for col in results.columns
    }

    model_col = next(
        (
            columns[key]
            for key in ["algorithm", "model", "classifier"]
            if key in columns
        ),
        None
    )

    metric_candidates = [
        "accuracy", "precision", "recall", "f1",
        "roc_auc", "roc-auc"
    ]

    metric_cols = [
        columns[key]
        for key in metric_candidates
        if key in columns
    ]

    if model_col and metric_cols:
        selected_metric = st.selectbox(
            "Choose a metric to compare",
            metric_cols
        )

        chart_data = results[[model_col, selected_metric]].copy()
        chart_data[selected_metric] = pd.to_numeric(
            chart_data[selected_metric],
            errors="coerce"
        )
        chart_data = chart_data.dropna()

        fig = px.bar(
            chart_data,
            x=model_col,
            y=selected_metric,
            title=f"Model Comparison — {selected_metric}",
            labels={
                model_col: "Model",
                selected_metric: selected_metric
            }
        )

        fig.update_layout(
            xaxis_tickangle=-35,
            yaxis_range=[0, 1.05]
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(
            "The results file does not contain the expected "
            "model and classification metric columns. "
            "The table above is still available."
        )


# --------------------------------------------------
# 7. ABOUT PAGE
# --------------------------------------------------
elif page == "About Project":

    st.title("About Our Project")

    st.markdown("""
    ### Machine Learning Capstone — Team 13

    This application provides an interactive interface
    for the team's machine learning work.

    **Current interactive model**
    - Task: Customer churn classification
    - Dataset: Telco Customer Churn
    - Model: Saved Part A classification pipeline

    **Application features**
    - Interactive customer input form
    - Prediction from a saved model
    - Churn probability when supported
    - Model evaluation table and comparison chart

    **Technologies**
    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    - Plotly
    """)

    st.caption(
        "Predictions are intended for educational "
        "demonstration and are not guarantees."
    )
