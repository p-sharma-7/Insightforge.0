import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.components.data_ingestion import DataIngestion
from src.components.data_transformtion import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utlis import load_object, save_object

# Page layout
st.set_page_config(page_icon=":bar_chart:")

# Title of the app
st.title('Automated Machine Learning Prediction App')

# Step 1: Upload dataset
st.header("Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())
    
    # Displaying basic dataset information
    st.subheader("Dataset Information")
    st.write(f"Number of rows: {data.shape[0]}")
    st.write(f"Number of columns: {data.shape[1]}")

    st.subheader("Dataset Statistics")
    st.write(data.describe())

    # Step 2: Select target column
    st.header("Step 2: Select Target Column")
    target_column = st.selectbox("Select the target column", data.columns)

    # Step 3: Choose between manual or automatic model selection
    st.header("Step 3: Choose Model Selection Method")
    model_selection_method = st.radio("Select model selection method", ('Automatic', 'Manual'))

    selected_models = None
    if model_selection_method == 'Manual':
        st.subheader("Select Models to Train")
        selected_models = st.multiselect(
            "Choose models", 
            ['Random Forest', 'Decision Tree', 'Gradient Boosting', 'Linear Regression', 'XGBRegressor', 'CatBoost Regressor', 'AdaBoost Regressor']
        )

    if st.button("Start Training"):
        with st.spinner('Training models...'):
            # Data ingestion
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initate_data_ingestion(data)

            # Data transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_path=train_data_path, 
                test_path=test_data_path, 
                target_column_name=target_column
            )

            # Model training
            model_trainer = ModelTrainer()
            best_model_name, best_model_score, mse, r2 = model_trainer.train_model(train_arr, test_arr)

            # Display results
            st.success('Training completed!')
            st.write(f"Best Model: {best_model_name}")
            st.write(f"Model Score: {best_model_score}")
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"R2 Score: {r2}")

            # Save best model
            save_object(file_path=model_trainer.model_trainer_config.trained_model_file_path, obj=model_trainer)

            # Plot feature importance if available
            if hasattr(model_trainer, 'feature_importances_'):
                st.subheader("Feature Importances")
                feature_importances = model_trainer.feature_importances_
                features = data.drop(columns=[target_column]).columns
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': feature_importances
                }).sort_values(by='Importance', ascending=False)

                fig, ax = plt.subplots()
                sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
                st.pyplot(fig)

# Step 4: Make predictions
st.header("Step 4: Make Predictions")

if st.button("Load Model"):
    try:
        model = load_object(file_path=model_trainer.model_trainer_config.trained_model_file_path)
        st.success('Model loaded successfully!')
    except Exception as e:
        st.error(f"Error loading model: {e}")

input_data = {}
if uploaded_file is not None:
    st.subheader("Input Data for Prediction")
    for col in data.drop(columns=[target_column]).columns:
        input_data[col] = st.text_input(f"Enter value for {col}")

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        preprocessor = load_object(preprocessor_path)
        input_transformed = preprocessor.transform(input_df)

        prediction = model.predict(input_transformed)
        st.write(f"Prediction: {prediction[0]}")
