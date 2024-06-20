import os
import sys
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

import pandas as pd 
import numpy as np

st.set_page_config(page_title="Home page", page_icon=":bar_chart:", layout="wide")


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
st.write('---')
