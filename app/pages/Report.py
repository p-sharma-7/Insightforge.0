import streamlit as st
import pandas as pd
import dtale
from sklearn.datasets import load_diabetes
from dtale.app import build_app
from streamlit.server.server import Server
import os
import threading

# Page layout
st.set_page_config(page_title="Report", page_icon=":bar_chart:", layout="wide")

st.title("Interactive Data Exploration with D-Tale")
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

st.markdown('')
st.markdown('')
st.subheader("Example Datasets")

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

# Function to run D-Tale server in the background
def run_dtale(data):
    d = dtale.show(data, subprocess=False)
    d.open_browser()

# D-Tale Integration
if st.session_state.data is not None:
    st.header('**Input DataFrame**')
    st.write(st.session_state.data)
    st.write('---')
    st.header('**Explore Data with D-Tale**')

    # Button to start D-Tale server and embed in the application
    if st.button("Start D-Tale"):
        # Run D-Tale in a separate thread
        threading.Thread(target=run_dtale, args=(st.session_state.data,)).start()
        
        # Display the D-Tale app in an iframe within the Streamlit app
        dtale_url = f"http://localhost:40000/dtale/main"  # D-Tale runs by default on port 40000
        st.markdown(f'<iframe src="{dtale_url}" width="100%" height="800"></iframe>', unsafe_allow_html=True)
else:
    st.warning("No dataset loaded. Please upload a file or select a dataset.")
import streamlit as st
import pandas as pd
import dtale
from sklearn.datasets import load_diabetes
from dtale.app import build_app
from streamlit.server.server import Server
import os
import threading

# Page layout
st.set_page_config(page_title="Report", page_icon=":bar_chart:", layout="wide")

st.title("Interactive Data Exploration with D-Tale")
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

st.markdown('')
st.markdown('')
st.subheader("Example Datasets")

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

# Function to run D-Tale server in the background
def run_dtale(data):
    d = dtale.show(data, subprocess=False)
    d.open_browser()

# D-Tale Integration
if st.session_state.data is not None:
    st.header('**Input DataFrame**')
    st.write(st.session_state.data)
    st.write('---')
    st.header('**Explore Data with D-Tale**')

    # Button to start D-Tale server and embed in the application
    if st.button("Start D-Tale"):
        # Run D-Tale in a separate thread
        threading.Thread(target=run_dtale, args=(st.session_state.data,)).start()
        
        # Display the D-Tale app in an iframe within the Streamlit app
        dtale_url = f"http://localhost:40000/dtale/main"  # D-Tale runs by default on port 40000
        st.markdown(f'<iframe src="{dtale_url}" width="100%" height="800"></iframe>', unsafe_allow_html=True)
else:
    st.warning("No dataset loaded. Please upload a file or select a dataset.")
