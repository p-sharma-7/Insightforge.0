import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.datasets import load_diabetes

#---------------Page layout------------------#

st.set_page_config(page_title="Prediction", page_icon=":bar_chart:", layout="wide")

st.title("Prediction")
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


#----------------dataset example-------------------------#

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

st.markdown('')
st.markdown('')
st.subheader("Example Datasets: ")

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



#----------------dataset example finished-------------------------#

#----------------prediction-------------------------#

# Data preprocessing and model training functions
def preprocess_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Keep only numerical features
    X = X.select_dtypes(include=[np.number])
    
    if X.empty:
        raise ValueError("The dataset does not contain any numerical features.")
    
    feature_names = X.columns
    
    # Standardize numerical variables
    ssc = StandardScaler()
    X = ssc.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test, ssc, feature_names

def get_model(algo):
    if algo == "Linear Regression":
        return LinearRegression()
    elif algo == "Ridge Regression":
        return Ridge()
    elif algo == "Lasso Regression":
        return Lasso()
    elif algo == "Decision Tree":
        return DecisionTreeRegressor()
    elif algo == "Random Forest":
        return RandomForestRegressor()
    elif algo == "Gradient Boosting":
        return GradientBoostingRegressor()
    elif algo == "Support Vector Regression":
        return SVR()

def tune_hyperparameters(model, X_train, y_train):
    param_grid = {}
    
    if isinstance(model, Ridge):
        param_grid = {'alpha': [0.1, 1.0, 10.0]}
    elif isinstance(model, Lasso):
        param_grid = {'alpha': [0.01, 0.1, 1.0]}
    elif isinstance(model, DecisionTreeRegressor):
        param_grid = {'max_depth': [3, 5, 10, None]}
    elif isinstance(model, RandomForestRegressor):
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None]}
    elif isinstance(model, GradientBoostingRegressor):
        param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    elif isinstance(model, SVR):
        param_grid = {'C': [0.1, 1.0, 10.0], 'epsilon': [0.01, 0.1, 0.2]}
    
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    else:
        return model

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        st.pyplot(plt)

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.tight_layout()
    st.pyplot(plt)

def identify_column_types(data):

    objective_cols = []
    numeric_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            objective_cols.append(col)
        else:
            numeric_cols.append(col)
    return objective_cols, numeric_cols

# Initialize session state
if 'trained' not in st.session_state:
    st.session_state['trained'] = False

# Main script
if data is not None:
    df = pd.read_csv(data)
    st.write(df.head())
    st.markdown('---')

    # Identify column types
    objective_cols, numeric_cols = identify_column_types(df)
    st.subheader("Column's Section: ")
    st.write("Select the column type you want to see")
    column_type = ["Categorical Columns","Numerical Columns"]
    column_type_choice = st.select_slider("",column_type)

    if column_type_choice == "Categorical Columns":
        st.write(objective_cols)
    if column_type_choice == "Numerical Columns":
        st.write(numeric_cols)

    st.success("NOTE: Please make sure that the target column is numerical.")
    target_col = st.selectbox('Select the target column', df.columns)
    
    # Let the user select specific algorithms to use
    algorithms = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "Support Vector Regression"]
    selected_algorithms = st.multiselect('Select algorithms to train', algorithms, default=algorithms)
    
    if st.button("Train Models"):
        try:
            X_train, X_test, y_train, y_test, ssc, feature_names = preprocess_data(df, target_col)
            
            r2_scores = []
            best_models = []
            
            for algo in selected_algorithms:
                model = get_model(algo)
                tuned_model = tune_hyperparameters(model, X_train, y_train)
                tuned_model.fit(X_train, y_train)
                y_pred = tuned_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                r2_scores.append(r2)
                best_models.append(tuned_model)
                
                st.subheader(f"Model: {algo}")
                st.write(f"R2 Score: {round(r2 * 100, 2)}%")
                plot_predictions(y_test, y_pred)
                plot_feature_importance(tuned_model, feature_names)
            
            best_algo_index = r2_scores.index(max(r2_scores))
            best_algo = selected_algorithms[best_algo_index]
            best_model = best_models[best_algo_index]
            
            st.subheader(f"The algorithm with the highest Accuracy is: {best_algo}")
            st.write(f"**R2 Score is = {round(r2_scores[best_algo_index], 4) * 100}%**")
            
            # Store model and scaler in session state
            st.session_state['best_model'] = best_model
            st.session_state['ssc'] = ssc
            st.session_state['feature_names'] = feature_names
            st.session_state['trained'] = True
                
        except ValueError as e:
            st.error(e)

# Prediction form
if st.session_state['trained']:
    st.subheader("Make Predictions with the Best Model")
    input_data = {}
    for feature in st.session_state['feature_names']:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)
    
    if st.button("Predict"):
        input_df = pd.DataFrame(input_data, index=[0])
        input_scaled = st.session_state['ssc'].transform(input_df)
        prediction = st.session_state['best_model'].predict(input_scaled)
        Best_model= st.session_state['best_model']
        st.write(f"Best Model: {Best_model}")
        st.write(f"Predicted value: {prediction[0]}")