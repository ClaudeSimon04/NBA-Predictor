import streamlit as st
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster

# Placeholder function for AI prediction
def predict_stat(player_name, opponent, stat_type, stat_value):
    """Mock AI prediction function."""
    return "Higher" if stat_value < 15 else "Lower"

# Streamlit UI Setup
st.title("NBA Player Performance Predictor")
st.write("Predict if a player will perform higher or lower than a given stat.")

# Fetch NBA teams
team_data = teams.get_teams()
teams_dict = {team["full_name"]: team["id"] for team in team_data}
team_list = sorted(teams_dict.keys())
team = st.selectbox("Select an NBA Team", team_list)

# Fetch players from the selected team using NBA API
def get_team_players(team_id):
    try:
        roster = commonteamroster.CommonTeamRoster(team_id=team_id)
        df = roster.get_data_frames()[0]
        return df['PLAYER'].tolist()
    except Exception as e:
        return []

# Get players for the selected team
team_id = teams_dict.get(team, None)
team_players = get_team_players(team_id)

# Handle case where no players are found
if not team_players:
    team_players = ["No players available"]

player = st.selectbox("Select a Player", team_players)

# Select Opponent Team
opponent = st.selectbox("Select the Opponent Team", team_list)

# User Input for Stat Prediction
stat_type = st.selectbox("Select Stat Type", ["Points", "Assists", "Rebounds", "Steals", "Blocks"])
stat_value = st.number_input("Enter Stat Value", min_value=0, max_value=100, step=1)

# Predict Button
if st.button("Predict Performance"):
    if "No players available" in player:
        st.write("No valid players for this team.")
    else:
        prediction = predict_stat(player, opponent, stat_type.upper(), stat_value)
        st.write(f"Prediction: {player} is likely to have a **{prediction}** {stat_type} than {stat_value}.")
