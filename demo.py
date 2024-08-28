import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures


# streamlit run c:/Users/Administrator/Downloads/enrollment1/app.py




# Load the trained model
model = joblib.load('bestmodel_for_prediction.pkl')

# Load the PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# Title
st.title("University Enrollment Prediction")

# Sidebar
st.sidebar.header("Input Features")

# Function to get user input
def get_user_input():
    start_year = st.sidebar.number_input('Registration Year', min_value=2005, max_value=2100, step=1)
    selected = st.sidebar.number_input('Selected Candidates (eg. 300)', min_value=0)
    capacity = st.sidebar.number_input('Holding Capacity (eg. 270)', min_value=0)
    course_popularity = st.sidebar.number_input('Course Popularity (eg. 300)', min_value=0.0, step=0.1)
    capacity_utilization = st.sidebar.number_input('Capacity Utilization (eg. 300)', min_value=0.0, max_value=1.0, step=0.01)
    registered_lag_1 = st.sidebar.number_input('Registered Lag 1 (eg. 300)', min_value=0)

    user_data = {
        'START_YEAR': start_year,
        'SELECTED': selected,
        'CAPACITY': capacity,
        'COURSE_POPULARITY': course_popularity,
        'CAPACITY_UTILIZATION': capacity_utilization,
        'REGISTERED_LAG_1': registered_lag_1
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
input_df = get_user_input()

# Display user input
st.subheader("User Input:")
st.write(input_df)

# Apply PolynomialFeatures transformation
input_poly = poly.fit_transform(input_df)

# Make prediction
prediction = model.predict(input_poly)

st.markdown(
    """
    <style>
    .stButton>button {
        display: inline-block;
        padding: 10px 20px;
        border: 1px solid #007BFF;
        border-radius: 5px;
        background-color: #007BFF;
        color: #FFFFFF;
        text-align: center;
        text-decoration: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #218838;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the button-like card
if st.button("Click to Predict"):
    st.write(f"Predicted Registered Students: {int(prediction[0])}")