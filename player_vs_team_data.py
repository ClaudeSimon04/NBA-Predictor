from nba_api.stats.endpoints import playergamelog
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Standardized stat mapping
STAT_MAPPING = {
    "POINTS": "PTS",
    "ASSISTS": "AST",
    "REBOUNDS": "REB",
    "STEALS": "STL",
    "BLOCKS": "BLK",
    "TURNOVERS": "TOV",
    "FIELDGOALPERCENT": "FG_PCT"
}

# Function to get player performance against each team
def get_player_vs_team_data(player_id, opponent, season='2024-25', last_n_games=5):
    """Fetch game logs for a player and calculate performance against a specific team, considering recent trends."""
    try:
        game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = game_log.get_data_frames()[0]
        
        # Print available columns for debugging
        print("Available columns in dataset:", df.columns.tolist())
        
        # Extract relevant columns
        df = df[['MATCHUP', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'GAME_DATE']]
        
        # Extract opponent team
        df['Opponent'] = df['MATCHUP'].apply(lambda x: x.split(' ')[-1])
        
        # Ensure date parsing works correctly
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
        df = df.sort_values(by='GAME_DATE', ascending=False)
        
        # Consider only the last N games against this opponent
        df = df[df['Opponent'].str.contains(opponent, case=False, na=False)].head(last_n_games)
        
        # Convert numeric columns to float and handle missing values
        numeric_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return df
    except Exception as e:
        print(f"Error fetching data for player {player_id} against {opponent}: {e}")
        return None

# Machine Learning Prediction Function
def train_and_predict(player_data, stat_type, stat_value):
    """Train a machine learning model to predict Higher or Lower outcome."""
    if player_data is None or player_data.empty:
        print("Not enough data to train the model.")
        return None
    
    # Print available stats before selecting stat_type
    print("Available stats in dataset:", player_data.columns.tolist())
    
    # Map stat type to correct column name
    stat_type = STAT_MAPPING.get(stat_type.upper(), stat_type.upper())
    if stat_type not in player_data.columns:
        print(f"Invalid stat type: {stat_type}. Available stats: {list(player_data.columns)}")
        return None
    
    # Preparing the dataset
    X = player_data[['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT']]
    y = (player_data[stat_type] > stat_value).astype(int)  # 1 = Higher, 0 = Lower
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make a prediction
    latest_game = X.iloc[-1:].values.reshape(1, -1)
    prediction = model.predict(latest_game)
    
    return "Higher" if prediction[0] == 1 else "Lower"

# Function to get user input
def get_user_input():
    """Prompt user for team, player, opponent team, and desired stat."""
    team = input("Enter the NBA team name: ")
    player = input(f"Enter a player from {team}: ")
    opponent = input(f"Enter the team {player} will play against: ")
    stat_input = input("Enter the stat you want to predict (e.g., '25 points', '10 assists'): ")
    
    # Parse user input
    try:
        stat_value = int(stat_input.split()[0])
        stat_type = stat_input.split()[1].upper()
    except (IndexError, ValueError):
        print("Invalid input. Please enter a number followed by the stat type.")
        return None
    
    return team, player, opponent, stat_type, stat_value

# Example usage
user_data = get_user_input()
if user_data:
    team, player, opponent, stat_type, stat_value = user_data
    player_id = 2544  # Placeholder, needs player ID lookup
    stats_vs_opponent = get_player_vs_team_data(player_id, opponent)
    
    if stats_vs_opponent is not None:
        prediction = train_and_predict(stats_vs_opponent, stat_type, stat_value)
        if prediction:
            print(f"Prediction: {player} is likely to have a {prediction} {stat_type} than {stat_value}, based on recent trends and AI model.")
