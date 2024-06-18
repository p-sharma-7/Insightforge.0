import os
import sys
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

import pandas as pd 
import numpy as np
from app.pages import EDA, Prediction

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

with st.sidebar:
            st.header("Data upload")
            uploaded_file = st.file_uploader("Choose a file")
'''
@dataclass
class DataIngestionConfig:
    Data_for_EDA : str = os.path.join('artifacts','data_for_EDA.csv')

@dataclass
class Insightforge:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        df=pd.DataFrame(uploaded_file)
        os.makedirs(os.path.dirname(self.ingestion_config.Data_for_EDA), exist_ok=True)
        df.to_csv(self.ingestion_config.Data_for_EDA, index=False, header=True)
        return df
'''

list_of_tabs = ["EDA", "Data visualization", "Prediction",'Dashboard']
tabs = st.tabs(list_of_tabs)

with tabs[0]:
    st.switch_page('pages/EDA.py')

with tabs[1]:
      st.switch_page('Data visualization.py')

with tabs[2]:
    st.switch_page('Prediction.py')

with tabs[3]:
       st.switch_page('Dashboard.py')



