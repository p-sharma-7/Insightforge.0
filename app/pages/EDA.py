import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_diabetes
#from pandas_profiling import ProfileReport
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report



#---------------Page layout------------------#

st.set_page_config(page_title="Exploratory Data Analysis", page_icon=":bar_chart:", layout="wide")

st.title("Exploratory Data Analysis")
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


#-----------------EDA------------------#
col1, col2, = st.columns(2)



if data is not None:
    df = pd.read_csv(data)
    df.dropna(how='any', inplace=True)
    df.drop_duplicates(keep='first', inplace=True)


    with col2:
        if st.checkbox('Show Full Dataset'):
            st.write(df)
        else:
            st.write(df.head())

    with col1:
        st.subheader("Data Information")
        st.markdown('---')
        st.write('Data Shape:', df.shape)
        st.write('Data Columns:', df.columns)
        st.write('Data Types:', df.dtypes)
        st.write('Data Missing Values:', df.isnull().sum())
        st.markdown('---')

    with col1:
        st.subheader("Pandas Profiling Report")
        if st.button("Generate Pandas Profiling Report"):
            pr= ProfileReport(df, explorative=True)
            st_profile_report(pr)
        

    with col2:
        col2.subheader("Descriptive statistics")
        st.write(df.describe(include='all'))
        st.markdown('---')
    
    
    with col2:
        col2.subheader("Value Counts")
        all_columns = df.columns.to_list()
        selected_columns = st.multiselect("Select a selected_columns_names for value counts", all_columns)
        # Check if any columns are selected
        if selected_columns:
            for column_name in selected_columns:
                st.write(df[column_name].value_counts())
        else:
            st.write("Please select at least one column.")
        st.markdown('---')
    
    with col2:
        col2.subheader("Coorelation Plot")
        st.success("")
        # Apply LabelEncoder to string columns
        df_encoded = df.copy()
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df_encoded[col] = le.fit_transform(df_encoded[col])
        # Plot correlation matrix using seaborn
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        st.pyplot(fig)
        st.markdown('---')
        
    with col2:
        col2.subheader("Pie Plot")
        st.success("NOTE: Converting the grouping smaller slices into others to aviod cluttering.")
        all_columns = df.columns.to_list()
        column_to_plot = st.selectbox("Select 1 Column for Pie Plot", all_columns)

        # Custom autopct function to format percentage text
        def custom_autopct(pct):
            return ('%.1f%%' % pct) if pct > 0 else ''
        
        fig, ax = plt.subplots(figsize=(8, 6))  # Create a new figure with specified size
        pie_data = df[column_to_plot].value_counts()

        # Threshold for grouping smaller slices into "Others"
        threshold = 0.05  # You can adjust this value as needed
        other_threshold = pie_data.sum() * threshold

        # Create a new Series for the pie chart
        pie_data_adjusted = pie_data.copy()
        pie_data_adjusted['Others'] = pie_data[pie_data < other_threshold].sum()
        pie_data_adjusted = pie_data_adjusted[pie_data_adjusted >= other_threshold]

        # Generate colors for the pie chart
        colors = plt.cm.Paired(range(len(pie_data_adjusted)))

        pie_data_adjusted.plot.pie(
            autopct=custom_autopct, 
            textprops={'fontsize': 10},  # Adjust fontsize for better readability
            ax=ax,
            colors=colors,  # Apply colors
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'},  # Enhance wedge appearance
            startangle=90  # Start the first wedge at 90 degrees
        )
        
        ax.set_ylabel('')  # Remove the default ylabel
        ax.set_title(f'Pie Chart of {column_to_plot}', fontsize=14)  # Set a title
        
        st.pyplot(fig)  # Display the figure in Streamlit

else:
    st.warning("No dataset loaded. Please upload a file or select a dataset.")