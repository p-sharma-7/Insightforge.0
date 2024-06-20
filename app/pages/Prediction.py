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

#---------------Page layout------------------#

st.set_page_config(page_title="Prediction", page_icon=":bar_chart:", layout="wide")

st.title("Prediction")
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
st.markdown('')
st.markdown('')
st.markdown('')
data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])

#-----------------pagelayout finished------------------#


#----------------data ingestion-------------------------#

# Define functions to load datasets
@st.cache_data
def load_diabetes_dataset():
    diabetes = load_diabetes()
    diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    diabetes_df['TARGET'] = diabetes.target
    return diabetes_df

@st.cache_data
def load_studentperformance_dataset():
    return pd.read_csv('https://raw.githubusercontent.com/p-sharma-7/Insightforge.0/main/artifacts/StudentsPerformance_dataset.csv')

@st.cache_data
def load_tips_dataset():
    return pd.read_csv('https://raw.githubusercontent.com/p-sharma-7/Insightforge.0/main/artifacts/tips_dataset.csv')

# Initialize session state to track selected dataset and checkboxes
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None
    st.session_state.data = None

if 'diabetes_checkbox' not in st.session_state:
    st.session_state.diabetes_checkbox = False

if 'studentperformance_checkbox' not in st.session_state:
    st.session_state.studentperformance_checkbox = False

if 'titanic_checkbox' not in st.session_state:
    st.session_state.titanic_checkbox = False

# Define function to reset other checkboxes
def reset_other_checkboxes(selected):
    st.session_state.diabetes_checkbox = selected == 'diabetes'
    st.session_state.studentperformance_checkbox = selected == 'studentperformance'
    st.session_state.titanic_checkbox = selected == 'titanic'

# Streamlit layout with three columns
cole1, cole2, cole3 = st.columns(3)

# Load diabetes dataset
with cole2:
    if st.checkbox("Use diabetes Dataset", key='diabetes_checkbox', on_change=reset_other_checkboxes, args=('diabetes',)):
        if st.session_state.diabetes_checkbox:
            st.session_state.selected_dataset = 'diabetes'
            st.session_state.data = load_diabetes_dataset()
            st.session_state.data.to_csv('artifacts/diabetes_dataset.csv', index=False)
            data='artifacts/diabetes_dataset.csv'

# Load student performance dataset
with cole1:
    if st.checkbox("Use studentperformance Dataset", key='studentperformance_checkbox', on_change=reset_other_checkboxes, args=('studentperformance',)):
        if st.session_state.studentperformance_checkbox:
            st.session_state.selected_dataset = 'studentperformance'
            st.session_state.data = load_studentperformance_dataset()
            st.session_state.data.to_csv('artifacts/studentperformance_dataset.csv', index=False)
            data='artifacts/studentperformance_dataset.csv'

# Load tips dataset
with cole3:
    if st.checkbox("Use tips Dataset", key='tips_checkbox', on_change=reset_other_checkboxes, args=('titanic',)):
        if st.session_state.titanic_checkbox:
            st.session_state.selected_dataset = 'titanic'
            st.session_state.data = load_tips_dataset()
            st.session_state.data.to_csv('artifacts/tips_dataset.csv', index=False)
            data='artifacts/tips_dataset.csv'



#----------------data ingestion finished-------------------------#

#----------------Prediction started-------------------------#





if data is not None:
    df = pd.read_csv(data)

    st.dataframe(df.head())
    if st.checkbox('Show Full Dataset'):
        st.write(df)
    
    columns = [feature for feature in df.columns if df[feature].dtypes != 'O']
    target_column_name = st.selectbox("Select Target Column", columns)

    # Data ingestion and data will save in the artifacts folder
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initate_data_ingestion(df)

    # Data transformation and processed data will save in the artifacts folder
    if target_column_name is not None:
        data_transformation = DataTransformation()
        train_array, test_array, preprocessor_file_path = data_transformation.initiate_data_transformation(train_data, test_data, target_column_name)
        
    # Algorithm Selection
    Algo_selection_type = ["Manual Algo Selection", "Automatic Algo Selection"]    
    st.subheader("Your Algorithm Preference?")
    algo_choice = st.select_slider("Slide To Change", Algo_selection_type)                

    if algo_choice == "Automatic Algo Selection":
        st.success("Using Automatic Algorithm Selection")
        model_trainer = ModelTrainer()
        best_model_name, best_model_score, mse, r2 = model_trainer.train_model(train_array, test_array)
        
        st.markdown("### Model Performance Metrics")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Best Model", best_model_name)
        with metrics_col2:
            st.metric("R2 Score", f"{r2:.2f}")
        with metrics_col3:
            st.metric("MSE", f"{mse:.2f}")


        # Visualizations
        st.markdown("### Predictions Visualization")
        predictions = PredictPipeline.predict(test_array)
        actuals = test_array[:, -1]

        fig, ax = plt.subplots()
        sns.scatterplot(x=actuals, y=predictions, ax=ax)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted Values")
        st.pyplot(fig)

    if algo_choice == "Manual Algo Selection":
        Algo = ["Linear Regression", "Ridge And Lasso", "Decision Tree"]
        st.subheader("Select The Algorithm!")
        selected_algo = st.selectbox("Select Algorithm", Algo)

        if selected_algo == "Linear Regression":
            st.success("Using Linear Regression")
            df_ml = df.copy()

            # Dropping High Cardinality column
            if "Job Title" in df_ml.columns:
                df_ml.drop("Job Title", axis=1, inplace=True)

            # Encode categorical columns
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

            pred_choice = st.radio(label="Select Here", options=["Get Predictions", "Show Description And Efficiency Of Model"])
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

            if pred_choice == "Show Description And Efficiency Of Model":
                # Evaluate the model
                st.markdown("### Model Performance Metrics")
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("R2 Score", f"{r2_score(y_test, y_pred) * 100:.2f} %")
                with metrics_col2:
                    st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
                with metrics_col3:
                    st.metric("RMSE", f"{mean_squared_error(y_test, y_pred) ** 0.5:.2f}")

                # Coefficients and intercept
                st.markdown("### Model Coefficients")
                coefficient = linear_regression_model.coef_
                intercept = linear_regression_model.intercept_
                st.write("Coefficients: ", coefficient)
                st.write("Intercept: ", intercept)

            if pred_choice == "Get Predictions":
                st.subheader("Enter the values for prediction:")

                user_input = []
                for col in df_ml.columns[:-1]:  # Exclude the target column
                    if col == "Gender":
                        user_val = st.selectbox(f"Select {col}:", options=["Male", "Female"])
                        user_input.append(1 if user_val == "Male" else 0)
                    elif col == "Education Level":
                        user_val = st.selectbox(f"Select {col}:", options=["Bachelors", "Masters", "PhD"])
                        user_input.append({"Bachelors": 0, "Masters": 1, "PhD": 2}[user_val])
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
else:
    st.warning("No dataset loaded. Please upload a file or select a dataset.")