import streamlit as st
import pandas as pd
import sweetviz as sv
from sklearn.datasets import load_diabetes
import base64
import os

# Page layout
st.set_page_config(page_title="Report", page_icon=":bar_chart:", layout="wide")

st.title("Interactive Data Exploration with Sweetviz")
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

# Function to generate and display Sweetviz report
def generate_sweetviz_report(df):
    report = sv.analyze(df)
    report_path = "sweetviz_report.html"
    report.show_html(filepath=report_path, open_browser=False)
    return report_path

# Function to create a download link for the Sweetviz report
def create_download_link(file_path, file_name="sweetviz_report.html"):
    with open(file_path, 'rb') as f:
        bytes_data = f.read()
    b64 = base64.b64encode(bytes_data).decode()
    href = f'<a href="data:file/html;base64,{b64}" download="{file_name}">Download Sweetviz Report</a>'
    return href

# Sweetviz Report Integration
if st.session_state.data is not None:
    st.header('**Input DataFrame**')
    st.write(st.session_state.data)
    st.write('---')
    st.header('**Sweetviz Report**')

    # Button to generate Sweetviz report and display in the app
    if st.button("Generate Sweetviz Report"):
        st.success("This may take a while")
        report_path = generate_sweetviz_report(st.session_state.data)

        # Display a message to the user that the report was generated successfully
        st.success("Sweetviz report generated successfully!")

        # Provide the download button
        st.markdown(create_download_link(report_path), unsafe_allow_html=True)

        # Optional: Ask user if they want to display the report inside the app
        if st.checkbox("Display Report in App (May Cause Performance Issues)"):
            try:
                # Read the report and display it in an iframe
                with open(report_path, 'r', encoding='utf-8') as file:
                    report_html = file.read()
                    st.components.v1.html(report_html, width=1000, height=800, scrolling=True)
            except Exception as e:
                st.error(f"Error displaying report: {str(e)}")
else:
    st.warning("No dataset loaded. Please upload a file or select a dataset.")
