import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import threading
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(page_title="Advanced Data Analytics Dashboard", layout="wide")

# Title and description
st.title("Advanced Data Analytics Dashboard")
st.markdown("Upload a CSV dataset to explore, visualize, and even model your data interactively.")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        # ---- Basic Statistics (with threading) ----
        st.write("### Basic Statistics")
        stats_container = st.empty()

        def compute_summary(data, container):
            container.write(data.describe())

        thread = threading.Thread(target=compute_summary, args=(df, stats_container))
        thread.start()

        # ---- Visualization Options ----
        st.write("### Data Visualization")
        columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        viz_choice = st.selectbox("Select Visualization Type", ["Histogram", "Correlation Matrix", "Scatterplot", "Boxplot"])

        if viz_choice == "Histogram":
            selected_column = st.selectbox("Select column", columns)
            if selected_column:
                fig, ax = plt.subplots()
                sns.histplot(df[selected_column], kde=True, ax=ax)
                ax.set_title(f"Distribution of {selected_column}")
                st.pyplot(fig)

        elif viz_choice == "Correlation Matrix":
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots()
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Matrix")
                st.pyplot(fig)
            else:
                st.warning("Not enough numeric columns for correlation matrix.")

        elif viz_choice == "Scatterplot":
            if len(numeric_cols) > 1:
                x_axis = st.selectbox("X-axis", numeric_cols)
                y_axis = st.selectbox("Y-axis", numeric_cols)
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
                ax.set_title(f"{x_axis} vs {y_axis}")
                st.pyplot(fig)

        elif viz_choice == "Boxplot":
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0 and len(numeric_cols) > 0:
                cat_col = st.selectbox("Categorical column", cat_cols)
                num_col = st.selectbox("Numeric column", numeric_cols)
                fig, ax = plt.subplots()
                sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
                ax.set_title(f"{num_col} by {cat_col}")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.warning("Need at least one categorical and one numeric column.")

        # ---- Data Filtering ----
        st.write("### Filter Data")
        filter_column = st.selectbox("Select column to filter", columns)
        unique_values = df[filter_column].unique()
        selected_value = st.selectbox("Select value to filter", unique_values)
        filtered_df = df[df[filter_column] == selected_value]
        st.write(filtered_df)

        # Download button for filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download filtered data", data=csv, file_name="filtered_data.csv", mime="text/csv")

        # ---- Machine Learning: Simple Linear Regression ----
        st.write("### 🔮 Simple Linear Regression Model")
        if len(numeric_cols) > 1:
            target = st.selectbox("Select target column", numeric_cols)
            features = st.multiselect("Select feature columns", [c for c in numeric_cols if c != target])

            if target and features:
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                model = LinearRegression().fit(X_train, y_train)
                score = model.score(X_test, y_test)
                st.write(f"Model R² Score: {score:.2f}")
        else:
            st.info("Not enough numeric columns for regression.")

    except Exception as e:
        st.error(f"Error loading file: {e}")
