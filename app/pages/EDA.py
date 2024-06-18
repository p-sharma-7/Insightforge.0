import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
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

#---------------------------------#
# Page layout

st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

st.title("Exploratory Data Analysis")
list_of_tabs = ['Home',"EDA", "Data visualization", "Prediction"]
tabs = st.tabs(list_of_tabs)
with tabs[0]:
        st.page_link('Insightforge.py', label='Home', use_container_width=True)
with tabs[1]:
        st.page_link('pages/EDA.py', label='EDA', use_container_width=True)
with tabs[2]:
        st.page_link('pages/Data_visualization.py', label='Data visualization', use_container_width=True)
with tabs[3]:
        st.page_link('pages/Prediction.py', label='Prediction', use_container_width=True)


data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
if st.button("Use diabetes Dataset"):
    diabetes = load_diabetes()
    diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    diabetes_df['TARGET'] = diabetes.target
    diabetes_df.to_csv('diabetes_dataset.csv', index=False)
    data = 'diabetes_dataset.csv'

if st.button('Student Performance Dataset'):
    data= pd.read_csv("artifacts\StudentsPerformance_dataset.csv")

col1, col2, = st.columns(2)

#pagelayout finished
#---------------------------------#    


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
        col1.subheader("Variables in the dataset")
        all_columns = df.columns.to_list()
        st.write("These are the columns ",all_columns)

    with col2:
        col2.subheader("Summary")
        st.write(df.describe(include='all'))
    
    
    with col1:
        all_columns = df.columns.to_list()
        selected_columns = st.multiselect("Select Columns for display", all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)
    
    with col2:
        col2.subheader("Value Counts")
        all_columns = df.columns.to_list()
        selected_column = st.multiselect("Select a selected_columns_names for value counts", all_columns)
        for i in range(len(selected_column)):
            st.write(df[i].value_counts())
    
    with col1:
        col1.subheader("Coorelation Plot")
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
        
    with col2:
        col2.subheader("Pie Plot")
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
