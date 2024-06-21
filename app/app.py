import os
import sys
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

import pandas as pd 
import numpy as np
from streamlit_lottie import st_lottie
import requests


st.set_page_config(page_title="Home page", page_icon=":bar_chart:", layout="wide")


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
st.write('---')

# Title of the page
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        margin-top: -50px; /* Adjust the value as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<h1 class="centered-title">📋 Description</h1>', unsafe_allow_html=True)

def load_lottieurl(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

lottie_url = "https://lottie.host/91b19868-9056-42f7-98d7-139be0898573/0dhqnbrexr.json"
lottie_json = load_lottieurl(lottie_url)
st_lottie(lottie_json,height=150,key="analysis")

# Layout with two columns
st.markdown(
    """
    <style>
    .no-wrap {
        text-align: center;
        margin-top: -100px; /* Adjust the value as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    '<h3 class="no-wrap">Welcome to our <i>Streamlit-based Application</i> for <i>Deep Data Analysis and Forecasting</i>!</h3>',
    unsafe_allow_html=True
)

col1,col2,col3 = st.columns([1,2,3])
# Description
with col3:
    st.markdown("""
    *🔍 Exploratory Data Analysis (EDA):*
    - Gain insights through detailed statistical summaries and charts.

    *📊 Visualization:*
    - Create interactive charts and graphs to visualize data relationships and trends.
    """)
with col2:
    st.markdown("""
    *🔮 Prediction:*
    - Get your predictions via multiple machine learning models based 
                on your desired inputs.

    *📑 Automatic Reporting:*
    - Generate detailed reports with Pandas Profiling for easy 
                sharing and interpretation.
    """)

st.markdown(
    """
    <style>
    .no-wrap {
        text-align: center;
        margin-top: -10px; /* Adjust the value as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    '<h5 class="no-wrap">Our application offers a  USER FRIENDLY-INTERFACE,  DYNAMIC VISUALIZATION  and  POWERFULL FORCASTING POSSIBILITIES  making it an essential tool for data analysts and everyday users. </h5>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .no-wrap {
        text-align: center;
        margin-top: -20px; /* Adjust the value as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    '<h3 class="no-wrap">✨ *Start exploring your data today!* ✨</h3>',
    unsafe_allow_html=True
)
