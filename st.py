import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("ODI_Match_info.csv")

# Keep only the necessary columns
columns_to_keep = ['team1', 'team2', 'winner']
df = df[columns_to_keep]

# Create a dictionary that maps each team to a unique code
teams = {
    'India': 0,
    'England': 1,
    'New Zealand': 2,
    'Australia': 3,
    'Sri Lanka': 4,
    'South Africa': 5,
    'Bangladesh': 6,
    'Pakistan': 7,
    'Nepal': 8,
    'Afghanistan': 9,
    'West Indies': 10,
    'Scotland': 11,
    'United Arab Emirates': 12,
    'Oman': 13,
    'Netherlands': 14,
    'Zimbabwe': 15,
    'United States of America': 16,
    'Ireland': 17,
    'Canada': 18,
    'Namibia': 19,
    'Jersey': 20,
    'Papua New Guinea': 21,
    'Hong Kong': 22,
    'Kenya': 23,
    'Africa XI': 24,
    'Bermuda': 25,
    'Asia XI':26,
    'ICC World XI':27,
    'NAN':29
}

# Map the teams to their codes
df['team1'] = df['team1'].map(teams)
df['team2'] = df['team2'].map(teams)
df['winner'] = df['winner'].map(teams)

# Fill NaN values in the 'winner' column
df['winner'] = df['winner'].fillna(29)

# Split the dataframe into features and target
x = df.iloc[:,:2]
y = df.iloc[:,2]

# Train a random forest classifier with 15 trees
model = RandomForestClassifier(n_estimators=15)
model.fit(x, y)

# Create a mapping from codes back to team names
team_names = {v: k for k, v in teams.items()}

# Create Streamlit app
st.title('Cricket Match Winner Prediction')

# User inputs
team1 = st.selectbox('Select Team 1', list(teams.keys()))
team2 = st.selectbox('Select Team 2', list(teams.keys()))

if st.button('Predict Winner'):
    # Prepare data for prediction
    prediction_data = [[teams[team1], teams[team2]]]

    # Make prediction
    prediction = model.predict(prediction_data)

    # Display prediction
    st.write(f'The predicted winner is: {team_names[int(prediction[0])]}')
