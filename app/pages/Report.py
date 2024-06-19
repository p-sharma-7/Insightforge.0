import numpy as np
import pandas as pd
import streamlit as st
#from pandas_profiling import ProfileReport
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Web App Title
st.set_page_config(page_title="Exploratory Data Analysis", page_icon=":bar_chart:", layout="wide")
st.markdown('''
# **Complete EDA Report**
            
This is the **Complete EDA Report** created in Streamlit using the **pandas-profiling** library.

---
''')
coly1, coly2, coly3, coly4, coly5 = st.columns(5)
#list_of_tabs = ['Home',"EDA", "Data visualization", "Prediction"]
#tabs = st.tabs(list_of_tabs)
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

# Upload CSV data

uploaded_file = st.file_uploader("Upload a Dataset", type=["csv", "txt"])

# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache_data
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Example data
        @st.cache_data
        def load_data():
            a = pd.DataFrame(
                np.random.rand(100, 5),
                columns=['a', 'b', 'c', 'd', 'e']
            )
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)