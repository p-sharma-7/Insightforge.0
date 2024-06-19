import streamlit as st
import os
import sys

# Core Packages
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

# EDA Packages
import pandas as pd 
import numpy as np 

# Data Visualization Packages
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

# Encoding and ML Packages
from sklearn.preprocessing import LabelEncoder

#Custom Packages
from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformtion import DataTransformation
from src.components.data_transformtion import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
from src.components.predict_pipeline import PredictPipeline
from sklearn.datasets import load_diabetes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.title("Machine Learning Prediction Model")
st.write('---')
coly1, coly2, coly3, coly4, coly5 = st.columns(5)

with coly1:
        st.page_link('Insightforge.py', label='Home page', use_container_width=True)
        
with coly2:
        st.page_link('pages/EDA.py', label='EDA', use_container_width=True)
with coly3:
        st.page_link('pages/Data_visualization.py', label='Data visualization', use_container_width=True)
with coly4:
        st.page_link('pages/Prediction.py', label='Prediction', use_container_width=True)
with coly5:
        st.page_link('pages/Report.py', label='Report page', use_container_width=True)


data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])

if st.button("Use diabetes Dataset"):
        diabetes = load_diabetes()
        diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        diabetes_df['TARGET'] = diabetes.target
        diabetes_df.to_csv('diabetes_dataset.csv', index=False)
        data = 'diabetes_dataset.csv'

if data is not None:
            df = pd.read_csv(data)

            st.dataframe(df.head())
            if st.checkbox('Show Full Dataset'):
                st.write(df)
            
            columns= [feature for feature in df.columns if df[feature].dtypes != 'O']
            target_column_name = st.selectbox("Select Target Column", columns)

            #data_ingestion and data will save in the artifacts folder
            data_ingestion = DataIngestion()
            train_data, test_data = data_ingestion.initate_data_ingestion(df)


            #data transformation and processed data will save in the artifacts folder
            if target_column_name is not None:
                data_transformation = DataTransformation()
                train_array, test_array, preprocessor_file_path = data_transformation.initiate_data_transformation(train_data, test_data, target_column_name)
                

            #Algorithm Selection
            Algo_selection_type = ["Manual Algo Selection", "Automatic Algo Selection"]    
            st.subheader("Your Algorithm Preference ?")
            algo_choice = st.select_slider("Slide To Change",Algo_selection_type)                


            if algo_choice == "Automatic Algo Selection":

                st.success("Using Automatic Algorithm Selection")
                model_trainer = ModelTrainer()
                best_model_name,best_model_score,mse,r2 = model_trainer.train_model(train_array, test_array)
                st.write(f"Best Model Name: {best_model_name}")
                st.write(f"Best Model Score: {[best_model_score]}")
                st.write(f"Mean Squared Error: {mse}")
                st.write(f"R2 Score: {r2}")



            if algo_choice == "Manual Algo Selection":
                Algo = ["Linear Regression", "Ridge And Lasso", "Decision Tree"]
                st.subheader("Select The Algorithm ! ")
                selected_algo = st.selectbox("Select Algorithm",Algo)

                if selected_algo == "Linear Regression":
                    st.success("Using Linear Regression")
                    df_ml = df.copy()

                    # dropping High Cardinality column
                    if "Job Title" in df_ml.columns:
                        df_ml.drop("Job Title", axis=1, inplace=True)

                    # Encode categorical columns
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    for col in df_ml.select_dtypes(include=['object']).columns:
                        df_ml[col] = le.fit_transform(df_ml[col])

                    # Split data into predictors and target
                    X = df_ml.iloc[:, :-1]   # predictors
                    y = df_ml.iloc[:, -1]    # target

                    # Standardize the data
                    from sklearn.preprocessing import StandardScaler
                    ssc = StandardScaler()
                    X = ssc.fit_transform(X)

                    # Split the data into training and testing sets
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                    # Train the linear regression model
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

                    linear_regression_model = LinearRegression()
                    linear_regression_model.fit(X_train, y_train)

                    # Model prediction on test data
                    y_pred = linear_regression_model.predict(X_test)

                    pred_choice = st.radio(label="Select Here",options= ["Get Predictions", "Show Description And Efficiency Of Model"])
                    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>',unsafe_allow_html=True)

                    if pred_choice == "Show Description And Efficiency Of Model":
                        # Evaluate the model
                        st.write(f'Accuracy of the model: {round(r2_score(y_test, y_pred), 4) * 100} %')
                        st.write(f'Mean absolute error: {round(mean_absolute_error(y_test, y_pred), 2)}')
                        st.write(f'Mean squared error: {round(mean_squared_error(y_test, y_pred), 2) ** 0.5}')

                        # Coefficients and intercept
                        coefficient = linear_regression_model.coef_
                        intercept = linear_regression_model.intercept_
                        st.write("Coefficients: ", coefficient)
                        st.write("Intercept: ", intercept)

                    if pred_choice == "Get Predictions":
                        st.subheader("Enter the values for prediction:")

                        # Mapping for gender and education level
                        gender_map = {"Male": 1, "Female": 0}
                        education_map = {"Bachelors": 0, "Masters": 1, "PhD": 2}

                        user_input = []
                        for col in df_ml.columns[:-1]:  # Exclude the target column
                            if col == "Gender":
                                user_val = st.selectbox(f"Select {col}:", options=list(gender_map.keys()))
                                user_input.append(gender_map[user_val])
                            elif col == "Education Level":
                                user_val = st.selectbox(f"Select {col}:", options=list(education_map.keys()))
                                user_input.append(education_map[user_val])
                            elif df_ml[col].dtype == 'object':
                                user_val = st.text_input(f"Enter {col}:")
                                user_input.append(le.transform([user_val])[0])
                            else:
                                user_val = st.number_input(f"Enter {col}:", value=0)
                                user_input.append(user_val)
                        
                        # Convert user input to dataframe
                        input_df = pd.DataFrame([user_input], columns=df_ml.columns[:-1])

                        # Standardize user input
                        input_df = ssc.transform(input_df)

                        # Make prediction
                        prediction = linear_regression_model.predict(input_df)
                        
                        st.write("**Prediction:**", prediction[0])
