# important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Page Layout
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="InsightForge", page_icon=":bar_chart:",layout="wide"
)
st.title("Data Visualization")
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



def main():
    '''Data Visualization'''
    data = st.file_uploader("Upload Dataset",type=["csv"])
    if data is not None:

        # Read the uploaded file
        df = pd.read_csv(data)
        df_uncleaned = df.copy()

        # Drop rows with any missing values
        df.dropna(how='any', inplace=True)

        # Drop duplicate rows, keeping the first occurrence
        df.drop_duplicates(keep='first', inplace=True)

        # Displaying Dataset
        st.header("Dataset Section")
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

        st.header("Column's Section")
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

if __name__ == '__main__':
    main()