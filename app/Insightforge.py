import os
import sys
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

import pandas as pd 
import numpy as np

st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

def main():

    list_of_tabs = ['Home',"EDA", "Data visualization", "Prediction",'Dashboard']
    tabs = st.tabs(list_of_tabs)

    with tabs[0]:
        st.page_link('Insightforge.py', label='Home', use_container_width=True)
    with tabs[1]:
        st.page_link('pages/EDA.py', label='EDA', use_container_width=True)
    with tabs[2]:
        st.page_link('pages/Data_visualization.py', label='Data visualization', use_container_width=True)
    with tabs[3]:
        st.page_link('pages/Prediction.py', label='Prediction', use_container_width=True)
    with tabs[4]:
        st.page_link('pages/Dashboard.py', label='Dashboard', use_container_width=True)

        

if __name__ == '__main__':
    main()

