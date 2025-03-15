import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load Dataset
df=pd.read_csv("Artifacts/cleaned_data.csv")

# Load the trained model
try:
    model = load_model("Artifacts/ANN_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the preprocessor
try:
    with open('Artifacts/preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
except Exception as e:
    st.error(f"Error loading preprocessor: {e}")
    st.stop()

# Title and logo of the app
st.title("BankChurn AI Predictor")

# Sidebar input for user features
st.sidebar.header("Enter Details of Customer:")
user_input = {}
st.sidebar.markdown(
    "<p style='color: red; font-weight: bold;'>Note: Use 1 for Yes/True and 0 for No/False</p>",
    unsafe_allow_html=True
)

# Define numerical and categorical features
num_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
cat_features = ['Geography', 'Gender', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'AgeGroups', 'BalanceCategory']
feature_names = num_features + cat_features

# Create sidebar inputs with better defaults
user_input['CreditScore'] = st.sidebar.number_input("Enter Credit Score", min_value=300, max_value=850, value=650)
user_input['Age'] = st.sidebar.number_input("Enter Age", min_value=18, max_value=100, value=30)
user_input['Balance'] = st.sidebar.number_input("Enter Balance", min_value=0.0, value=50000.0)
user_input['EstimatedSalary'] = st.sidebar.number_input("Enter Estimated Salary", min_value=0.0, value=50000.0)

for feature in cat_features:
    user_input[feature] = st.sidebar.selectbox(f"Select {feature}", df[feature].unique())

# Convert user input to a DataFrame
input_df = pd.DataFrame([user_input])

# Handle missing values
input_df.fillna(0, inplace=True)

# Predict button
if st.sidebar.button("Predict"):
    try:
        # Apply preprocessing (fixed error)
        input_df = preprocessor.transform(input_df)

        
        # Get prediction (ANN outputs probability)
        probability = model.predict(input_df)[0][0]  # Extract probability

        # Display prediction result
        st.subheader("Prediction Result")
        if probability < 0.5:
            st.write("The customer is **not likely to churn.** 😊")
        else:
            st.write("The customer is **likely to churn.** ⚠️")

        
        # Display input features for clarity
        st.subheader("Based on Your Following Inputs:")
        st.write(pd.DataFrame([user_input]))
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer
st.sidebar.markdown("---")  # Horizontal line
st.sidebar.write("**App and Model Developed by**: Muhammad Waqas")
st.sidebar.write("Gmail: waqasliaqat630@gmail.com")
st.sidebar.write("Contact: +92 3097829808")

# Footer instructions
st.write("Use the sidebar to input values and click 'Predict' to get the results.")
