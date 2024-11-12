import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf

# Datasets and their URLs
datasets = {
    "Iris Dataset": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    "Titanic Dataset": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "Boston Housing Dataset": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
    "Diabetes Dataset": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "Wine Quality Dataset": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "Heart Disease Dataset": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "Breast Cancer Wisconsin Dataset": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    "California Housing Dataset": "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv",
    "Mall Customers Dataset": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mall_customers.csv",
    "Student Performance Dataset": "https://archive.ics.uci.edu/ml/machine-learning-databases/student/student-mat.csv"
}

# Function to load dataset
@st.cache_data
def load_dataset(url):
    return pd.read_csv(url)

# App title
st.title('Machine Learning Basics App')
st.write("""
This app introduces users to basic machine learning concepts using preloaded datasets. Select a dataset, explore it, and apply simple machine learning models to understand the basics.
""")

# Select a dataset
selected_dataset = st.selectbox("Choose a Dataset", list(datasets.keys()))

# Load and display dataset
if selected_dataset:
    data = load_dataset(datasets[selected_dataset])
    st.write(f"Dataset Preview - {selected_dataset}:")
    st.write(data.head())

    # Display dataset details
    if st.checkbox("Show Dataset Info"):
        st.write(data.describe())
        st.write(data.info())

    # Train a basic model
    st.write("### Train a Model on this Dataset")

    # Select model type
    model_type = st.selectbox("Select Model Type", ["Linear Regression", "Decision Tree Classifier", "Random Forest", "SVM", "K-Means Clustering"])

    # Linear Regression for Regression Datasets
    if model_type == "Linear Regression" and selected_dataset in ["Boston Housing Dataset", "Diabetes Dataset"]:
        features = st.multiselect("Select Features", data.columns.tolist())
        target = st.selectbox("Select Target", data.columns.tolist())
        
        if features and target:
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            st.write("Model Coefficients:", model.coef_)
            st.write("Intercept:", model.intercept_)
            st.write("Mean Squared Error:", mse)

            # Plot actual vs predicted values
            plt.figure(figsize=(10, 5))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Actual vs Predicted Values")
            st.pyplot(plt)

    # Classification Models for Classification Datasets
    elif model_type in ["Decision Tree Classifier", "Random Forest", "SVM"] and selected_dataset in ["Iris Dataset", "Titanic Dataset", "Heart Disease Dataset", "Breast Cancer Wisconsin Dataset"]:
        features = st.multiselect("Select Features", data.columns.tolist())
        target = st.selectbox("Select Target", data.columns.tolist())

        if features and target:
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_type == "Decision Tree Classifier":
                model = DecisionTreeClassifier()
            elif model_type == "Random Forest":
                model = RandomForestClassifier(n_estimators=100)
            elif model_type == "SVM":
                model = SVC(kernel='linear')

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.write("Model Accuracy:", accuracy)

            # Plot feature importance if applicable
            if model_type in ["Decision Tree Classifier", "Random Forest"]:
                feature_importances = model.feature_importances_
                sns.barplot(x=features, y=feature_importances)
                plt.title("Feature Importances")
                st.pyplot(plt)

            # Confusion Matrix and Classification Report
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            st.write(cm)
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))

    # K-Means Clustering for Unsupervised Learning
    elif model_type == "K-Means Clustering" and selected_dataset in ["Mall Customers Dataset", "Student Performance Dataset"]:
        features = st.multiselect("Select Features", data.columns.tolist())
        if features:
            X = data[features]
            n_clusters = st.slider("Number of Clusters", 2, 10, value=3)
            model = KMeans(n_clusters=n_clusters)
            data['Cluster'] = model.fit_predict(X)

            st.write(data.head())

            # Plot clusters
            plt.figure(figsize=(10, 5))
            sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=data['Cluster'], palette='viridis')
            plt.title("K-Means Clustering")
            st.pyplot(plt)

    # Real-Time Prediction
    if st.checkbox("Make a Prediction") and model_type in ["Linear Regression", "Decision Tree Classifier", "Random Forest", "SVM"]:
        st.write("### Make a Real-Time Prediction")
        user_data = {}
        for feature in features:
            user_data[feature] = st.number_input(f"Input value for {feature}", float(X[feature].mean()))
        input_df = pd.DataFrame([user_data])
        prediction = model.predict(input_df)
        st.write("Prediction:", prediction)

    # Save and Load Model
    if st.button("Save Model") and model_type in ["Linear Regression", "Decision Tree Classifier", "Random Forest", "SVM"]:
        joblib.dump(model, 'trained_model.pkl')
        st.success("Model saved successfully!")

    if st.button("Load Model"):
        loaded_model = joblib.load('trained_model.pkl')
        st.success("Model loaded successfully!")

st.write("### Conclusion")
st.write("You've now trained a basic machine learning model on your chosen dataset and explored how different models like Linear Regression, Decision Trees, Random Forest, SVM, and K-Means Clustering work. Experiment further with different features, target variables, and datasets!")
