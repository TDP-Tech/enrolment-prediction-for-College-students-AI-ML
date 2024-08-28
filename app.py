import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures

# Load the trained model
model = joblib.load('bestmodel_for_prediction.pkl')

# Load the PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# Title
st.title("University Enrollment Prediction")

# Sidebar
st.sidebar.header("Input Features")

# Function to preprocess data and extract the start year
def preprocess_data(data):
    # Extract the start year from the year range format
    data['START_YEAR'] = data['YEAR'].apply(lambda x: int(x.split('/')[0]))
    return data

# Function to get user input
def get_user_input(data):
    # Define the range of years for the selectbox
    current_years = list(range(2020, 2116))

    # Entry year
    current_year = st.sidebar.selectbox('Entry Year', current_years)
    
    # Course selection
    course = st.sidebar.selectbox('Course', data['COURSE'].unique())
    
    # Selected candidates
    selected = st.sidebar.number_input('Selected Candidates (eg. 300)', min_value=0)
    
    # Holding capacity
    capacity = st.sidebar.number_input('Holding Capacity (eg. 270)', min_value=0)
    
    # Calculate course popularity as the mean number of registered students
    course_popularity = data.groupby('COURSE')['REGISTERED'].mean().reset_index()
    course_popularity.columns = ['COURSE', 'COURSE_POPULARITY']
    
    # Merge the popularity feature back into the original DataFrame
    data = pd.merge(data, course_popularity, on='COURSE', how='left')
    
    # Calculate the capacity utilization rate
    capacity_utilization = selected / capacity if capacity != 0 else 0
    
    # Create lag features for the number of registered students
    data = data.sort_values(by=['COURSE', 'START_YEAR'])
    data['REGISTERED_LAG_1'] = data.groupby('COURSE')['REGISTERED'].shift(1)
    data = data.fillna(0)
    
    # Assume the current year and course to calculate the lag feature
    registered_lag_1 = data[(data['COURSE'] == course) & (data['START_YEAR'] == current_year - 1)]['REGISTERED'].values
    registered_lag_1 = registered_lag_1[0] if len(registered_lag_1) > 0 else 0
    
    user_data = {
        'START_YEAR': current_year,
        'SELECTED': selected,
        'CAPACITY': capacity,
        'COURSE_POPULARITY': data[data['COURSE'] == course]['COURSE_POPULARITY'].values[0],
        'CAPACITY_UTILIZATION': capacity_utilization,
        'REGISTERED_LAG_1': registered_lag_1
    }

    features = pd.DataFrame(user_data, index=[0])
    return features, course

# Assume 'data' is your DataFrame that contains historical data
data = pd.read_csv('ENROLMENT.csv')  # Replace with the actual path to your historical data

# Preprocess data to extract the start year
data = preprocess_data(data)

# Get user input
input_df, selected_course = get_user_input(data)

# Display historical data for the selected course
st.write(f"Historical Data for Course: {selected_course}")
historical_data = data[data['COURSE'] == selected_course]
st.dataframe(historical_data)

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
    .dataframe {
        width: 100% !important;
        overflow: auto;
        display: block;
    }
    .dataframe th, .dataframe td {
        text-align: center !important;
        vertical-align: middle !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the button-like card
if st.button("Click to Predict"):
    st.write(f"Predicted Students that can be Registered this Year =  {int(prediction[0])}")
