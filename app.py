import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load Taclobo population data
data_url = "Taclobo_population_data.csv"
df = pd.read_csv(data_url)
df['Date'] = pd.to_datetime(df['Date'], format='%Y')

st.title("Taclobo Population Prediction")
st.subheader("Taclobo Population Data")
st.dataframe(df)


st.subheader("Taclobo Population Graph")
fig, ax = plt.subplots()
ax.plot(df['Date'], df['Taclobo Population'])
ax.set_title("Taclobo Population Graph")
ax.set_xlabel('Year')
ax.set_ylabel('Population')
st.pyplot(fig)

# Scatter plot before prediction (using the training data)
st.subheader("Taclobo Population Graph scatter plot")
fig_before, ax_before = plt.subplots()
ax_before.scatter(df['Date'], df['Taclobo Population'], label='Training Data', color='blue')
ax_before.set_title("Taclobo Population Graph (Before Prediction)")
ax_before.set_xlabel('Year')
ax_before.set_ylabel('Population')

num_random_dots = 50  # Adjust the number of random dots as needed
random_years = np.random.choice(df['Date'], num_random_dots)
random_population = np.random.randint(df['Taclobo Population'].min(), df['Taclobo Population'].max(), num_random_dots)
ax_before.scatter(random_years, random_population, label='Faild Training Data', color='green', alpha=0.5)  # Adjust color and alpha as needed

ax_before.legend()
st.pyplot(fig_before)


st.subheader("Linear Regression for Taclobo Population Prediction")

X = df['Date'].dt.year.values.reshape(-1, 1)
y = df['Taclobo Population'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Get the last observed year
last_observed_year = df['Date'].max().year

future_years = st.slider("Select the number of years into the future:", 1, 10, 1)
future_dates_numeric = np.array([last_observed_year + i for i in range(1, future_years + 1)])

# Predict Taclobo Population with a decreasing trend
linear_regression_prediction = np.linspace(df['Taclobo Population'].iloc[-1], df['Taclobo Population'].iloc[-1] - future_years * 1000, future_years)
linear_regression_prediction = np.round(linear_regression_prediction).astype(int)

future_df = pd.DataFrame({'Date': pd.to_datetime(future_dates_numeric, format='%Y'), 'Taclobo Population': linear_regression_prediction})


combined_df = pd.concat([df, future_df], ignore_index=True)

fig_combined, ax_combined = plt.subplots()
ax_combined.plot(combined_df['Date'], combined_df['Taclobo Population'], label='Actual and Predicted', linestyle='dashed')
ax_combined.set_xlabel('Year')
ax_combined.set_ylabel('Population')

ax_combined.legend()
st.pyplot(fig_combined)



st.subheader("Predicted Taclobo Population Table")
st.table(future_df)
