from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd

# Define the season
season = '2024-25'

# Fetch player statistics
player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season)

# Convert the data to a pandas DataFrame
df = player_stats.get_data_frames()[0]

# Display the first few rows of the DataFrame
print(df.head())

# Save the DataFrame to a CSV file
df.to_csv('nba_player_stats_2024_2025.csv', index=False)