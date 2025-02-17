import streamlit as st
import pandas as pd
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import commonteamroster, playergamelog
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Fetch Player Game Logs
def get_player_game_logs(player_id, season='2023-24'):
    """Fetch recent game logs for a player."""
    try:
        game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = game_log.get_data_frames()[0]
        
        # ✅ Debugging: Print DataFrame to check if data exists
        print(df.head())  

        if df.empty:
            print(f"⚠️ No game log data found for player ID {player_id} in season {season}")

        return df[['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT']]
    except Exception as e:
        print(f"❌ Error fetching game logs for player ID {player_id}: {e}")
        return None

# Fetch Player ID
def get_player_id(player_name):
    """Find the player ID from their name."""
    all_players = players.get_players()
    for player in all_players:
        if player["full_name"].lower() == player_name.lower():
            return player["id"]
    return None

# AI Prediction Logic Using Deep Learning
def predict_stat(player_name, opponent, stat_type, stat_value, player_id):
    """Train a deep learning model to predict if the player will exceed the stat_value."""
    player_data = get_player_game_logs(player_id)
    
    if player_data is None or player_data.empty:
        return "Not enough data to predict."

    # Preparing the dataset
    X = player_data[['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT']]
    y = (player_data[stat_type] > stat_value).astype(int)  # 1 = Higher, 0 = Lower

    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define the deep learning model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model with more epochs
    model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)

    # Make a prediction for the next game
    latest_game = scaler.transform(X.iloc[-1:].values.reshape(1, -1))
    prediction = model.predict(latest_game)

    return "Higher" if prediction[0][0] > 0.5 else "Lower"

# Streamlit UI Setup
st.title("NBA Player Performance Predictor")
st.write("Predict if a player will perform higher or lower than a given stat.")
st.write("**By Claude Simon**")  # Added author name

# Fetch NBA teams
team_data = teams.get_teams()
teams_dict = {team["full_name"]: team["id"] for team in team_data}
team_list = sorted(teams_dict.keys())

team = st.selectbox("Select an NBA Team", team_list)

# Fetch players from the selected team using NBA API
def get_team_players(team_id):
    """Fetches players from the selected team"""
    try:
        roster = commonteamroster.CommonTeamRoster(team_id=team_id)
        df = roster.get_data_frames()[0]
        return df['PLAYER'].tolist()
    except Exception as e:
        st.error(f"Error fetching players: {e}")
        return []

# Get players for the selected team
team_id = teams_dict.get(team, None)
team_players = get_team_players(team_id)

# Handle case where no players are found
if not team_players:
    team_players = ["No players available"]

player = st.selectbox("Select a Player", team_players)

# Select Opponent Team
opponent = st.selectbox("Select the Opponent Team", [t for t in team_list if t != team])

# User Input for Stat Prediction
stat_type = st.selectbox("Select Stat Type", ["PTS", "AST", "REB", "STL", "BLK"])
stat_value = st.number_input("Enter Stat Value", min_value=0, max_value=100, step=1)

# Predict Button
if st.button("Predict Performance"):
    player_id = get_player_id(player)
    print(f"🔍 Fetching data for {player} (Player ID: {player_id})")  # ✅ Debugging
    if player_id:
        prediction = predict_stat(player, opponent, stat_type.upper(), stat_value, player_id)
        st.write(f"Prediction: {player} is likely to have a **{prediction}** {stat_type} than {stat_value}.")
    else:
        st.write("Error: Player ID not found.")
