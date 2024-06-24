import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


#---------------Page layout------------------#

st.set_page_config(page_title="Prediction", page_icon=":bar_chart:", layout="wide")
st.title("Machine Learning Prediction Model")
st.write('---')
coly1, coly2, coly3, coly4, coly5 = st.columns(5)

with coly1:
        st.page_link('app.py', label='Home page', use_container_width=True)
        
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


if data is not None:
            df = pd.read_csv(data)
            df.dropna(how='any', inplace=True)
            df.drop_duplicates(keep='first', inplace=True)
            st.dataframe(df.head())

            if st.checkbox('Show Full Dataset'):
                st.write(df)

            Algo_selection_type = ["Manual Algo Selection", "Automatic Algo Selection"]    
            st.subheader("Your Algorithm Preference ?")
            algo_choice = st.select_slider("Slide To Change", Algo_selection_type)

            if algo_choice == "Manual Algo Selection":
                Algo = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree"]
                st.subheader("Select The Algorithm ! ")
                selected_algo = st.selectbox("Select Algorithm", Algo)
                df_ml = df.copy()

                def preprocess_data(df):
                    if "Job Title" in df_ml.columns:
                        df_ml.drop("Job Title", axis=1, inplace=True)
                    from sklearn.preprocessing import LabelEncoder    
                    le = LabelEncoder()
                    for col in df_ml.select_dtypes(include=['object']).columns:
                        df_ml[col] = le.fit_transform(df_ml[col])
                    X = df_ml.iloc[:, :-1]  # predictors
                    y = df_ml.iloc[:, -1]   # target
                    from sklearn.preprocessing import StandardScaler
                    ssc = StandardScaler()
                    X = ssc.fit_transform(X)
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                    return X_train, X_test, y_train, y_test, ssc, le

                def get_model(algo):
                    from sklearn.linear_model import LinearRegression, Ridge, Lasso
                    from sklearn.tree import DecisionTreeRegressor
                    if algo == "Linear Regression":
                        return LinearRegression()
                    elif algo == "Ridge Regression":
                        return Ridge(alpha=1.0)
                    elif algo == "Lasso Regression":
                        return Lasso(alpha=0.1)
                    elif algo == "Decision Tree":
                        return DecisionTreeRegressor() 

                if selected_algo in ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree"]:
                    st.success(f"Using {selected_algo}")
                    X_train, X_test, y_train, y_test, ssc, le = preprocess_data(df)
                    model = get_model(selected_algo)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    pred_choice = st.radio(label="Select Here", options=["Get Predictions", "Show Description And Efficiency Of Model"])
                    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
                
                    if pred_choice == "Show Description And Efficiency Of Model":
                        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                        st.write(f'Accuracy of the model: {round(r2_score(y_test, y_pred), 4) * 100} %')
                        st.write(f'Mean absolute error: {round(mean_absolute_error(y_test, y_pred), 2)}')
                        st.write(f'Mean squared error: {round(mean_squared_error(y_test, y_pred), 2) ** 0.5}')
                        if selected_algo != "Decision Tree":
                            coefficient = model.coef_
                            intercept = model.intercept_
                            st.write("Coefficients: ", coefficient)
                            st.write("Intercept: ", intercept)

                    if pred_choice == "Get Predictions":
                        st.subheader("Enter the values for prediction:")
                        gender_map = {"Male": 1, "Female": 0}
                        education_map = {"Bachelors": 0, "Masters": 1, "PhD": 2}
                        user_input = []
                        for col in df_ml.columns[:-1]:
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
                        
                        input_df = pd.DataFrame([user_input], columns=df_ml.columns[:-1])
                        input_df = ssc.transform(input_df)
                        prediction = model.predict(input_df)

                        if st.button("**Get Salary**"):
                            st.write("**Prediction:**", prediction[0])

            elif algo_choice == "Automatic Algo Selection":
                st.success("I Am Ready With The Best ML Model For The Dataset Provided !!")
                df_Ml = df.copy()

                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import LabelEncoder, StandardScaler
                from sklearn.linear_model import LinearRegression, Ridge, Lasso
                from sklearn.tree import DecisionTreeRegressor
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

                def preprocess_data_copied(df_Ml):
                    if "Job Title" in df_Ml.columns:
                        df_Ml.drop("Job Title", axis=1, inplace=True)
                    le = LabelEncoder()
                    for col in df_Ml.select_dtypes(include=['object']).columns:
                        df_Ml[col] = le.fit_transform(df_Ml[col])
                    X = df_Ml.iloc[:, :-1]  # predictors
                    y = df_Ml.iloc[:, -1]   # target
                    ssc = StandardScaler()
                    X = ssc.fit_transform(X)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                    return X_train, X_test, y_train, y_test, ssc, le
                
                def get_model(algo):
                    if algo == "Linear Regression":
                        return LinearRegression()
                    elif algo == "Ridge Regression":
                        return Ridge(alpha=1.0)
                    elif algo == "Lasso Regression":
                        return Lasso(alpha=0.1)
                    elif algo == "Decision Tree":
                        return DecisionTreeRegressor() 
                    
                # Define algorithms to test
                algorithms = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree"]
                # Preprocess the data
                X_train, X_test, y_train, y_test, ssc, le = preprocess_data_copied(df_Ml)
                # Store R2 scores in a list
                r2_scores = []
                # Loop through each algorithm
                for algo in algorithms:                    
                    # Get the model
                    model = get_model(algo)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    # Calculate R2 score
                    r2 = r2_score(y_test, y_pred)
                    # Store the R2 score in the list
                    r2_scores.append(r2)

                # Find the best algorithm based on R2 score
                best_algo_index = r2_scores.index(max(r2_scores))
                best_algo = algorithms[best_algo_index]

                # Display the best algorithm
                if st.button("Get The Model"):
                    st.subheader(f"The algorithm with the highest Accuracy is: {best_algo}")
                    st.write(f"**R2 Score is = {round(r2_scores[best_algo_index], 4) * 100}%**")
                
                
