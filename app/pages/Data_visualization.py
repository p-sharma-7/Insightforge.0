# important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_diabetes



#---------------Page layout------------------#

st.set_page_config(page_title="Data visualization", page_icon=":bar_chart:", layout="wide")

st.title("Data visualization")
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

st.markdown('')
st.markdown('')
st.subheader("Example Datasets: ")

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

#----------------data visualization started-------------------------#


def main():
    '''Data Visualization'''
    if data is not None:

        # Read the uploaded file
        df = pd.read_csv(data)
        df_uncleaned = df.copy()

        # Drop rows with any missing values
        df.dropna(how='any', inplace=True)

        # Drop duplicate rows, keeping the first occurrence
        df.drop_duplicates(keep='first', inplace=True)

        # Displaying Dataset
        st.subheader("Dataset Section")
        st.write('Select whether you want to see the cleaned or uncleaned dataset')
        dataset_type = ["Cleaned Dataset","Uncleaned Dataset"]
        dataset_type_choice = st.select_slider("",dataset_type)

        if dataset_type_choice == "Cleaned Dataset":
            st.write("Data Shape")
            st.write(df.shape)
            st.write(df)
        if dataset_type_choice == "Uncleaned Dataset":
            st.write("Data Shape")
            st.write(df_uncleaned.shape)
            st.write(df_uncleaned)

        # Differentiating the columns on the basis of their data types
        def identify_column_types(data):

            objective_cols = []
            numeric_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    objective_cols.append(col)
                else:
                    numeric_cols.append(col)
            return objective_cols, numeric_cols
        
        objective_cols, numeric_cols = identify_column_types(df) # passing the cleaned dataset

        st.subheader("Column's Section: ")
        st.write("Select the type of columns you want to visualize")
        column_type = ["Categorical Columns","Numerical Columns"]
        column_type_choice = st.select_slider("",column_type)

        if column_type_choice == "Categorical Columns":
                st.write(objective_cols)
            
        if column_type_choice == "Numerical Columns":
                st.write(numeric_cols)

        
        st.header("Plot's Section")
        tab1,tab2,tab3 = st.tabs(["Univariate Plots","Summary Plots","Bivariate Plots"])
        # plot_type_choice = st.radio("",plot_type)
        # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>',unsafe_allow_html=True)

        
        with tab1:
            st.subheader("Univariate Plots")
            st.markdown(
                '''
                **This plot contains those types of plots which take only one argument as input !** \n
                '''
            )
            col1,col2 = st.columns(2)
            columns_uni = st.selectbox("Select Column", df.columns)
            if df[columns_uni].dtype == 'object':
                with col1:
                    col1.subheader("HISTOGRAM")
                    sns.histplot(x=columns_uni, data=df)  # Correct usage of sns.histplot
                    plt.xlabel(columns_uni)
                    st.pyplot()
                with col2:
                    col2.subheader("BAR CHART")
                    sns.countplot(x=columns_uni, data=df)  # Correct usage of sns.countplot
                    plt.xlabel(columns_uni)
                    st.pyplot()
                with col1:
                    col1.subheader("PIE CHART")
                    category_counts = df[columns_uni].value_counts()
                    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%') 
                    st.pyplot()

                with col2:
                    col2.subheader("BOX PLOT")
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=df[columns_uni])
                    st.pyplot()

            else:
                with col1:
                    col1.subheader("BOX PLOT")
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=df[columns_uni])
                    st.pyplot()

                with col2:
                    col2.subheader("DENSITY PLOT")
                    sns.kdeplot(df[columns_uni], shade=True)
                    st.pyplot()

                with col1:
                    col1.subheader("HISTOGRAM")
                    sns.histplot(x=columns_uni, data=df)  # Correct usage of sns.histplot
                    plt.xlabel(columns_uni)
                    st.pyplot()

                with col2:
                    col2.subheader("BAR CHART")
                    sns.countplot(x=columns_uni, data=df)  # Correct usage of sns.countplot
                    plt.xlabel(columns_uni)
                    st.pyplot()

                with col1:
                    col1.subheader("PIE CHART")
                    category_counts = df[columns_uni].value_counts()
                    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%') 
                    st.pyplot()


        with tab2:
            st.subheader("Summary Plots")
            st.markdown(
                '''
                **This plot is used to describe a short summary of the dataset which do not requires any columns to be selected !**
                '''
            )
            col1,col2 = st.columns(2)
            with col1:
                col1.subheader("Heat Map")            
                # Apply LabelEncoder to string columns
                df_encode = df.copy()
                le = LabelEncoder()
                for col in df.select_dtypes(include=['object']).columns:
                    df_encode[col] = le.fit_transform(df_encode[col])
                # Plotting heatmap matrix using seaborn
                fig, ax = plt.subplots(figsize=(15, 15))
                sns.heatmap(df_encode.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
                st.pyplot(fig)

            with col2:
                col2.subheader("Pair Plot")
                # Create the pair plot
                fig, ax = plt.subplots(figsize=(15, 15))
                sns.pairplot(df) # Create a pair plot for all columns in the DataFrame
                st.pyplot()
                

        with tab3:
            st.subheader("Bivariate Plots")
            st.markdown(
                '''
                **This plot contains those types of plots which can take two argument as input for Data comparison !**
                '''
            )
            col1,col2 = st.columns(2)
            x_column = st.selectbox("Select X Column", df.columns)
            y_column = st.selectbox("Select Y Column", df.columns)

            with col1:
                col1.subheader("SCATTER PLOT")
                sns.scatterplot(x=df[x_column], y=df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                st.pyplot()

            with col2:
                col2.subheader("LINE PLOT")
                sns.lineplot(x=df[x_column], y=df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                st.pyplot()
                
            with col1:
                col1.subheader("AREA PLOT")
                plt.fill_between(df[x_column], df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                st.pyplot()

            with col2:
                col2.subheader("VIOLIN PLOT")
                plt.figure(figsize=(10, 6))
                sns.violinplot(x=df[x_column], y=df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                st.pyplot()
    else:
        st.warning("No dataset loaded. Please upload a file or select a dataset.")

if __name__ == '__main__':
    main()