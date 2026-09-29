import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

st.set_page_config(page_title="Insurance Charge Prediction", layout="wide")

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data()
def load_data():
    try:
        df = pd.read_csv("insurance.csv")
        return df
    except FileNotFoundError:
        st.error("❌ ERROR: 'insurance.csv' not found!")
        return None


# ------------------------------
# PREPROCESS DATA
# ------------------------------
def preprocess_data(df):
    df = df.copy()
    df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
    df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
    df.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)
    return df


# ------------------------------
# TRAIN MODEL
# ------------------------------
def train_model(df):
    X = df.drop(columns=["charges"])
    Y = df["charges"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, Y_train)

    return model, X_train, X_test, Y_train, Y_test


# ------------------------------
# MODEL EVALUATION
# ------------------------------
def evaluate_model(model, X_train, X_test, Y_train, Y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = metrics.r2_score(Y_train, y_pred_train)
    r2_test = metrics.r2_score(Y_test, y_pred_test)

    return r2_train, r2_test


# ------------------------------
# DATA INSIGHTS PAGE
# ------------------------------
def show_data_insights(df):
    st.header("📊 Data Insights & Visualization")
    st.write("### Dataset Shape")
    st.write(df.shape)

    st.write("### Statistical Summary")
    st.write(df.describe())

    # Plot 1: Correlation Heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

    # Distributions in 3 Columns
    st.subheader("Feature Distributions")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['age'], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("BMI Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['bmi'], kde=True, ax=ax)
        st.pyplot(fig)

    with col3:
        st.write("Charges Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['charges'], kde=True, ax=ax)
        st.pyplot(fig)

    # Countplots
    st.subheader("Category-wise Plots")
    colA, colB, colC = st.columns(3)

    with colA:
        st.write("Sex Count")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='sex', ax=ax)
        st.pyplot(fig)

    with colB:
        st.write("Smoker Count")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='smoker', ax=ax)
        st.pyplot(fig)

    with colC:
        st.write("Region Count")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='region', ax=ax)
        st.pyplot(fig)


# ------------------------------
# PREDICTION FUNCTION
# ------------------------------
def predict_cost(model, input_data):
    reshaped = np.asarray(input_data).reshape(1, -1)
    return model.predict(reshaped)[0]

# ------------------------------
# PREDICTION PAGE
# ------------------------------
def prediction_page(model):
    st.header("💰 Insurance Cost Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 100, 25)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        children = st.number_input("Children", 0, 10, 0)

    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

    # Encoding
    sex_code = 0 if sex == "male" else 1
    smoker_code = 0 if smoker == "yes" else 1
    region_code = ['southeast', 'southwest', 'northeast', 'northwest'].index(region)

    input_data = (age, sex_code, bmi, children, smoker_code,region_code)

    if st.button("Predict Cost"):
        cost = predict_cost(model, input_data)
        st.success(f"💡 **Predicted Insurance Charge: ₹ {cost:.2f}**")


# ------------------------------
# MAIN APP
# ------------------------------
def main():
    st.title("🏥 Insurance Charge Prediction System")

    df = load_data()
    if df is None:
        return

    df = preprocess_data(df)

    model, X_train, X_test, Y_train, Y_test = train_model(df)
    r2_train, r2_test = evaluate_model(model, X_train, X_test, Y_train, Y_test)

    # Sidebar Navigation
    menu = st.sidebar.radio(
        "📌 Navigation",
        ["Data Insights", "Model Performance", "Prediction"]
    )

    if menu == "Data Insights":
        show_data_insights(df)

    elif menu == "Model Performance":
        st.header("📈 Model Evaluation Report")
        st.write(f"### ✔ R² Score (Training): **{r2_train:.3f}**")
        st.write(f"### ✔ R² Score (Testing): **{r2_test:.3f}**")

    elif menu == "Prediction":
        prediction_page(model)


if __name__ == "__main__":
    main()

#streamlit run medical_insurance 