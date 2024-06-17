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

# Page Layout
st.set_page_config(
    page_title="InsightForge", page_icon=":bar_chart:"
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
            algo_choice = st.select_slider("Slide To Change", Algo_selection_type)

            if algo_choice == "Manual Algo Selection":
                Algo = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree"]
                st.subheader("Select The Algorithm ! ")
                selected_algo = st.selectbox("Select Algorithm", Algo)
                df_ml = df.copy()

                def preprocess_data(df):
                    if "Job Title" in df_ml.columns:
                        df_ml.drop("Job Title", axis=1, inplace=True)
                    from sklearn.preprocessing import LabelEncoder    
                    le = LabelEncoder()
                    for col in df_ml.select_dtypes(include=['object']).columns:
                        df_ml[col] = le.fit_transform(df_ml[col])
                    X = df_ml.iloc[:, :-1]  # predictors
                    y = df_ml.iloc[:, -1]   # target
                    from sklearn.preprocessing import StandardScaler
                    ssc = StandardScaler()
                    X = ssc.fit_transform(X)
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                    return X_train, X_test, y_train, y_test, ssc, le

                def get_model(algo):
                    from sklearn.linear_model import LinearRegression, Ridge, Lasso
                    from sklearn.tree import DecisionTreeRegressor
                    if algo == "Linear Regression":
                        return LinearRegression()
                    elif algo == "Ridge Regression":
                        return Ridge(alpha=1.0)
                    elif algo == "Lasso Regression":
                        return Lasso(alpha=0.1)
                    elif algo == "Decision Tree":
                        return DecisionTreeRegressor() 

                if selected_algo in ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree"]:
                    st.success(f"Using {selected_algo}")
                    X_train, X_test, y_train, y_test, ssc, le = preprocess_data(df)
                    model = get_model(selected_algo)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    pred_choice = st.radio(label="Select Here", options=["Get Predictions", "Show Description And Efficiency Of Model"])
                    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

                    if pred_choice == "Show Description And Efficiency Of Model":
                        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                        st.write(f'Accuracy of the model: {round(r2_score(y_test, y_pred), 4) * 100} %')
                        st.write(f'Mean absolute error: {round(mean_absolute_error(y_test, y_pred), 2)}')
                        st.write(f'Mean squared error: {round(mean_squared_error(y_test, y_pred), 2) ** 0.5}')
                        if selected_algo != "Decision Tree":
                            coefficient = model.coef_
                            intercept = model.intercept_
                            st.write("Coefficients: ", coefficient)
                            st.write("Intercept: ", intercept)

                    if pred_choice == "Get Predictions":
                        st.subheader("Enter the values for prediction:")
                        gender_map = {"Male": 1, "Female": 0}
                        education_map = {"Bachelors": 0, "Masters": 1, "PhD": 2}
                        user_input = []
                        for col in df_ml.columns[:-1]:
                            if col == "Gender":
                                user_val = st.selectbox(f"Select {col}:", options=list(gender_map.keys()))
                                user_input.append(gender_map[user_val])
                            elif col == "Education Level":
                                user_val = st.selectbox(f"Select {col}:", options=list(education_map.keys()))
                                user_input.append(education_map[user_val])
                            elif df_ml[col].dtype == 'object':
                                user_val = st.text_input(f"Enter {col}:")
                                user_input.append(le.transform([user_val])[0])
                            else:
                                user_val = st.number_input(f"Enter {col}:", value=0)
                                user_input.append(user_val)
                        
                        input_df = pd.DataFrame([user_input], columns=df_ml.columns[:-1])
                        input_df = ssc.transform(input_df)
                        prediction = model.predict(input_df)

                        if st.button("**Get Salary**"):
                            st.write("**Prediction:**", prediction[0])

            elif algo_choice == "Automatic Algo Selection":
                st.success("I Am Ready With The Best ML Model For The Dataset Provided !!")
                df_Ml = df.copy()

                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import LabelEncoder, StandardScaler
                from sklearn.linear_model import LinearRegression, Ridge, Lasso
                from sklearn.tree import DecisionTreeRegressor
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

                def preprocess_data_copied(df_Ml):
                    if "Job Title" in df_Ml.columns:
                        df_Ml.drop("Job Title", axis=1, inplace=True)
                    le = LabelEncoder()
                    for col in df_Ml.select_dtypes(include=['object']).columns:
                        df_Ml[col] = le.fit_transform(df_Ml[col])
                    X = df_Ml.iloc[:, :-1]  # predictors
                    y = df_Ml.iloc[:, -1]   # target
                    ssc = StandardScaler()
                    X = ssc.fit_transform(X)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                    return X_train, X_test, y_train, y_test, ssc, le
                
                def get_model(algo):
                    if algo == "Linear Regression":
                        return LinearRegression()
                    elif algo == "Ridge Regression":
                        return Ridge(alpha=1.0)
                    elif algo == "Lasso Regression":
                        return Lasso(alpha=0.1)
                    elif algo == "Decision Tree":
                        return DecisionTreeRegressor() 
                    
                # Define algorithms to test
                algorithms = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree"]
                # Preprocess the data
                X_train, X_test, y_train, y_test, ssc, le = preprocess_data_copied(df_Ml)
                # Store R2 scores in a list
                r2_scores = []
                # Loop through each algorithm
                for algo in algorithms:                    
                    # Get the model
                    model = get_model(algo)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    # Calculate R2 score
                    r2 = r2_score(y_test, y_pred)
                    # Store the R2 score in the list
                    r2_scores.append(r2)

                # Find the best algorithm based on R2 score
                best_algo_index = r2_scores.index(max(r2_scores))
                best_algo = algorithms[best_algo_index]

                # Display the best algorithm
                if st.button("Get The Model"):
                    st.subheader(f"The algorithm with the highest Accuracy is: {best_algo}")
                    st.write(f"**R2 Score is = {round(r2_scores[best_algo_index], 4) * 100}%**")
                
                

if __name__ == '__main__':
    main()