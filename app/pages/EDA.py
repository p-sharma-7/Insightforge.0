# Core Packages
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

# EDA Packages
import pandas as pd 
import numpy as np 

# Data Visualization Packages
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

# Encoding and ML Packages
from sklearn.preprocessing import LabelEncoder

st.title("Exploratory Data Analysis")

data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
        
if data is not None:
    df = pd.read_csv(data)
    df.dropna(how='any', inplace=True)
    df.drop_duplicates(keep='first', inplace=True)
    st.dataframe(df.head())
    
    if st.checkbox('Show Full Dataset'):
        st.write(df)
    
    if st.checkbox("Show Shape"):
        st.write(df.shape)
    
    if st.checkbox("Show Columns"):
        all_columns = df.columns.to_list()
        st.write(all_columns)
    
    if st.checkbox("Summary"):
        st.write(df.describe(include='all'))
    
    if st.checkbox("Show Selected Columns"):
        all_columns = df.columns.to_list()
        selected_columns = st.multiselect("Select Columns for display", all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)
    
    if st.checkbox("Show Value Counts"):
        all_columns = df.columns.to_list()
        selected_column = st.selectbox("Select a selected_columns_names for value counts", all_columns)
        st.write(df[selected_column].value_counts())
    
    if st.checkbox("Correlation Plot"):
        st.success("Building your correlation plot !")
        # Apply LabelEncoder to string columns
        df_encoded = df.copy()
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df_encoded[col] = le.fit_transform(df_encoded[col])
        # Plot correlation matrix using seaborn
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        st.pyplot(fig)
        
    if st.checkbox("Pie Plot"):
        all_columns = df.columns.to_list()
        column_to_plot = st.selectbox("Select 1 Column for Pie Plot", all_columns)
                
                # Custom autopct function to format percentage text
        def custom_autopct(pct):
            return ('%.1f%%' % pct) if pct > 0 else ''
                
        fig, ax = plt.subplots()  # Create a new figure
        df[column_to_plot].value_counts().plot.pie(
        autopct=custom_autopct, 
        textprops={'fontsize': 5},  # Change fontsize here
                    ax=ax
                )
        st.pyplot(fig)  # Pass the figure to st.pyplot()
