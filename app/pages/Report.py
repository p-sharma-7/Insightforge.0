import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.datasets import load_diabetes
import base64

# Page layout
st.set_page_config(page_title="Report", page_icon=":bar_chart:", layout="wide")

st.title("Pandas Profiling Report")
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

data = st.file_uploader("Upload a Dataset (CSV or TXT)", type=["csv", "txt"])

# Function to load datasets
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
            st.session_state.data = pd.read_csv('artifacts/diabetes_dataset.csv')

# Load student performance dataset
with cole1:
    if st.checkbox("Use studentperformance Dataset", key='studentperformance_checkbox', on_change=reset_other_checkboxes, args=('studentperformance',)):
        if st.session_state.studentperformance_checkbox:
            st.session_state.selected_dataset = 'studentperformance'
            st.session_state.data = load_studentperformance_dataset()
            st.session_state.data.to_csv('artifacts/studentperformance_dataset.csv', index=False)
            st.session_state.data = pd.read_csv('artifacts/studentperformance_dataset.csv')

# Load tips dataset
with cole3:
    if st.checkbox("Use tips Dataset", key='tips_checkbox', on_change=reset_other_checkboxes, args=('titanic',)):
        if st.session_state.titanic_checkbox:
            st.session_state.selected_dataset = 'titanic'
            st.session_state.data = load_tips_dataset()
            st.session_state.data.to_csv('artifacts/tips_dataset.csv', index=False)
            st.session_state.data = pd.read_csv('artifacts/tips_dataset.csv')

# Function to generate and download Pandas Profiling report
def download_profile_report(df):
    pr = ProfileReport(df, explorative=True)
    pr.to_file("pandas_profiling_report.html")
    with open("pandas_profiling_report.html", "rb") as f:
        bytes_data = f.read()
    b64 = base64.b64encode(bytes_data).decode()
    href = f'<a href="data:file/html;base64,{b64}" download="pandas_profiling_report.html">Download Pandas Profiling Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# Pandas Profiling Report
if st.session_state.data is not None:
    st.header('**Input DataFrame**')
    st.write(st.session_state.data)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(ProfileReport(st.session_state.data, explorative=True))  # Pass the DataFrame directly

    # Button to download report
    if st.button("Download Report"):
        download_profile_report(st.session_state.data)
else:
    st.warning("No dataset loaded. Please upload a file or select a dataset.")
