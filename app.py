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

# Encoding Packages
import json
import requests
from sklearn.preprocessing import LabelEncoder

# Animation Packages
#---

# Page Layout
st.set_page_config(
    page_title="InsightForge",page_icon=":bar_chart:"
)
page_bg_color = """
<style>
[data-testid="stAppViewContainer"] {
background-color: #b5eaf1;
}
[data-testid="stHeader"] {
background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True)

def main():
    """Semi Automated ML App with Streamlit"""
    
    activities = ["EDA", "Plots", "Predictions"]    
    choice = st.sidebar.selectbox("Select Activities", activities)

    if choice == 'EDA':
        st.title("Exploratory Data Analysis")

        data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
        if data is not None:
            # Read the uploaded file
            df = pd.read_csv(data)
            # Drop rows with any missing values
            df.dropna(how='any', inplace=True)
            # Drop duplicate rows, keeping the first occurrence
            df.drop_duplicates(keep='first', inplace=True)
            # Display the first few rows of the dataframe
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



    elif choice == 'Plots':
        st.title("Data Visualization")

        data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
        if data is not None:
            # Read the uploaded file
            df = pd.read_csv(data)
            # Drop rows with any missing values
            df.dropna(how='any', inplace=True)
            # Drop duplicate rows, keeping the first occurrence
            df.drop_duplicates(keep='first', inplace=True)
            # Display the first few rows of the dataframe
            st.dataframe(df.head())

            if st.checkbox('Show Full Dataset'):
                st.write(df)

            if st.checkbox("Show Value Counts"):
                all_columns = df.columns.to_list()
                selected_column = st.selectbox("Select a selected_columns_names for value counts", all_columns)
                st.write(df[selected_column].value_counts().plot(kind='bar'))
                st.pyplot()
    
            st.subheader("Select Plot Type")
            plot_choice = st.radio(label="Select Type Here",options= ["Regular Plots", "Comparing Plots"])
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>',unsafe_allow_html=True)

            if plot_choice == "Regular Plots":

                # Customizable Plot
                all_columns_names = df.columns.tolist()
                type_of_plot = st.selectbox("Select Type of Plot",['Histogram', 'Bar Plot', 'Box Plot','Pie Chart', 'Heatmap' 
                                                                   ,'Pair Plot', 'Density Plot'])
                selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)

                if st.button("Generate Plot"):
                    st.success(f"Generating Customizable Plot of {type_of_plot} for {selected_columns_names}")

                    # Plots 
                    if type_of_plot == 'Histogram':
                        plt.hist(df[selected_columns_names])
                        plt.xlabel(selected_columns_names)
                        st.pyplot()

                    if type_of_plot == 'Bar Plot':
                        count_data = df[selected_columns_names[0]].value_counts()
                        plt.bar(count_data.index, count_data.values)
                        plt.xlabel(selected_columns_names[0])
                        st.pyplot()

                    if type_of_plot == 'Box Plot':
                        st.write("NOTE: Do not work for the string contained columns !")
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(data=df[selected_columns_names])
                        st.pyplot()

                    if type_of_plot == 'Pie Chart':
                        plt.pie(df[selected_columns_names].value_counts(), labels=df[selected_columns_names].value_counts().index, autopct='%1.1f%%')
                        st.pyplot()

                    if type_of_plot == 'Heatmap': 
                        st.text("NOTE: No need to select the columns because its non functional in this plot!")
                        # Apply LabelEncoder to string columns
                        df_encode = df.copy()
                        le = LabelEncoder()
                        for col in df.select_dtypes(include=['object']).columns:
                            df_encode[col] = le.fit_transform(df_encode[col])
                        # Plotting heatmap matrix using seaborn
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(df_encode.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
                        st.pyplot(fig)

                    if type_of_plot == 'Pair Plot':
                        st.text("NOTE: No need to select the columns because its non functional in this plot!")
                        sns.pairplot(df)
                        st.pyplot()
                    
                    if type_of_plot == 'Density Plot':
                        st.text("NOTE: Do not Select the columns contains String !")
                        sns.kdeplot(df[selected_columns_names], shade=True)
                        st.pyplot() 

            if plot_choice == "Comparing Plots":

                # Comparing Plots
                type_of_plot_CP = st.selectbox("Select Type of Plot",["Scatter Plot","Line Plot","Area Plot","Violin Plot"])

                if type_of_plot_CP == 'Scatter Plot':
                    x_column = st.selectbox("Select X Column for Line Plot", df.columns)
                    y_column = st.selectbox("Select Y Column for Line Plot", df.columns)
                    plt.plot(df[x_column], df[y_column],color='blue', marker='o', linestyle='--', linewidth=1, markersize=5, )
                    plt.grid()
                    plt.xlabel(x_column)
                    plt.ylabel(y_column)
                    st.pyplot()
                    
                if type_of_plot_CP == 'Line Plot':
                    x_column = st.selectbox("Select X Column for Line Plot", df.columns)
                    y_column = st.selectbox("Select Y Column for Line Plot", df.columns)
                    plt.plot(df[x_column], df[y_column])
                    plt.xlabel(x_column)
                    plt.ylabel(y_column)
                    st.pyplot()
                
                if type_of_plot_CP == 'Area Plot':
                    x_column = st.selectbox("Select X Column for Area Plot", df.columns)
                    y_column = st.selectbox("Select Y Column for Area Plot", df.columns)
                    plt.fill_between(df[x_column], df[y_column])
                    plt.xlabel(x_column)
                    plt.ylabel(y_column)
                    st.pyplot()
                
                if type_of_plot_CP == 'Violin Plot':
                    x_column = st.selectbox("Select X Column for Violin Plot", df.columns)
                    y_column = st.selectbox("Select Y Column for Violin Plot", df.columns)
                    plt.figure(figsize=(10, 6))
                    sns.violinplot(x=df[x_column], y=df[y_column])
                    plt.xlabel(x_column)
                    plt.ylabel(y_column)
                    st.pyplot()   

    elif choice == 'Predictions':
        st.title("Machine Learning Prediction Model")
        data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
        if data is not None:
            # Read the uploaded file
            df = pd.read_csv(data)
            # Drop rows with any missing values
            df.dropna(how='any', inplace=True)
            # Drop duplicate rows, keeping the first occurrence
            df.drop_duplicates(keep='first', inplace=True)
            # Display the first few rows of the dataframe
            st.dataframe(df.head())

            if st.checkbox('Show Full Dataset'):
                st.write(df)

            Algo_selection_type = ["Manual Algo Selection", "Automatic Algo Selection"]    
            st.subheader("Your Algorithm Preference ?")
            algo_choice = st.select_slider("Slide To Change",Algo_selection_type)

            if algo_choice == "Automatic Algo Selection":
                st.write()

            if algo_choice == "Manual Algo Selection":
                Algo = ["Linear Regression", "Ridge And Lasso", "Decision Tree"]
                st.subheader("Select The Algorithm ! ")
                selected_algo = st.selectbox("Select Algorithm",Algo)

                if selected_algo == "Linear Regression":
                    st.success("Using Linear Regression")

                    # dropping High Cardinality column
                    df.drop("Job Title", axis=1, inplace=True)

                    # # Encode Gender
                    # from sklearn.preprocessing import LabelEncoder
                    # label_encoder = LabelEncoder()
                    # df["Gender"] = label_encoder.fit_transform(df["Gender"])
                    # df["Education Level"] = label_encoder.fit_transform(df["Education Level"])


            
       
            






if __name__ == '__main__':
    main()
