import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from model import X

# Load the trained model
model_filename = 'flight_delay_prediction_model_lightgbm_bayesian.pkl'
loaded_model = joblib.load(model_filename)


# Define preprocessing functions
def preprocess_sample_data(data, scaler=None):
    # Define distance bins and labels
    distance_bins = [0, 500, 1000, 1500, 2000, 2500, np.inf]
    distance_labels = ['0-500 km', '501-1000 km', '1001-1500 km', '1501-2000 km', '2001-2500 km', '2500+ km']

    # Simplify weather categories
    simplified_weather = {
        'Partly cloudy': 'Cloudy',
        'Cloudy': 'Cloudy',
        'Sunny': 'Sunny',
        'Patchy rain possible': 'Rain',
        'Clear': 'Clear',
        'Thundery outbreaks possible': 'Storm',
        'Light rain shower': 'Rain',
        'Moderate or heavy rain shower': 'Rain',
        'Overcast': 'Cloudy',
        'Showers': 'Rain',
        'Heavy rain': 'Rain',
        'Thunderstorm': 'Storm',
        'Drizzle': 'Rain',
        'Fog': 'Cloudy',
        'Mist': 'Cloudy',
        'Snow': 'Storm',
        'Sleet': 'Storm',
        'Hail': 'Storm',
        'Light freezing rain': 'Storm',
        'Freezing rain': 'Storm',
        'Ice pellets': 'Storm',
        'Light snow': 'Storm',
        'Moderate snow': 'Storm',
        'Heavy snow': 'Storm',
        'Blizzard': 'Storm',
        'Dust': 'Clear',  # Typically not significant in terms of weather impact
        'Sand': 'Clear',  # Similarly, often considered clear with reduced visibility
        'Smoke': 'Cloudy',  # Can obscure visibility and create hazy conditions
        'Volcanic ash': 'Storm',  # Can be severe and impact travel significantly
        'Squalls': 'Storm',  # Intense bursts of wind and precipitation
        'Gusts': 'Storm',  # Strong wind that can accompany storms
        'High winds': 'Storm',  # Can cause dangerous conditions
    }

    # Convert sample data to DataFrame
    df = pd.DataFrame([data])

    # Create distance categories
    df['Distance_Category'] = pd.cut(df['Distance'], bins=distance_bins, labels=distance_labels, right=False)
    df = df.drop(['Distance'], axis=1)

    # Simplify weather categories
    df['Simplified_Weather'] = df['Simplified_Weather'].map(simplified_weather).fillna('Unknown')

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['Airline', 'Simplified_Weather', 'Distance_Category', 'From', 'To'])

    # Ensure all columns are present in the training set
    for col in X.columns:
        if col not in df.columns:
            df[col] = 0
    df = df[X.columns]

    # Feature scaling
    if scaler:
        df_scaled = scaler.transform(df)
    else:
        df_scaled = df  # If no scaler, return raw data

    return df_scaled


# Adjust delay based on weather condition
def adjust_delay_based_on_weather(predicted_delay, weather_condition):
    # Adjustments based on weather
    weather_adjustments = {
        'Sunny': 0,  # No delay
        'Clear': 0,  # No delay
        'Cloudy': 5,  # Add 5 minutes
        'Rain': 15,  # Add 15 minutes
        'Storm': 30,  # Add 30 minutes
        'Unknown': 5  # Default adjustment if weather is unknown
    }
    adjustment = weather_adjustments.get(weather_condition, 5)  # Default to 5 minutes if not in dict
    adjusted_delay = predicted_delay + adjustment
    return adjusted_delay


# Adjust delay based on airport rating
def adjust_delay_based_on_airport_rating(adjusted_delay, airport_rating):
    # Adjustments based on airport rating
    if airport_rating > 0.8:
        rating_adjustment = -5  # Reduce delay by 5 minutes for high-rated airports
    elif airport_rating > 0.5:
        rating_adjustment = 0  # No adjustment for mid-rated airports
    else:
        rating_adjustment = 5  # Increase delay by 5 minutes for low-rated airports

    final_adjusted_delay = adjusted_delay + rating_adjustment
    return final_adjusted_delay


# Sample data
sample_data = {
    'From': "DEL", 'To': "MUM",
    'Departure Delay': 10,
    'Airline': "Air India",
    'Simplified_Weather': "Storm",
    'Distance': 600, 'Airline Rating': 0.88,
    'Airport Rating': 0.3,
}

# Load the scaler used during training
scaler_filename = 'scaleroptimize.pkl'
scaler = joblib.load(scaler_filename)

# Preprocess sample data
sample_df_scaled = preprocess_sample_data(sample_data, scaler=scaler)

# Predict delay using the trained model
predicted_delay = loaded_model.predict(sample_df_scaled)[0]

# Manually adjust the delay based on weather
adjusted_delay = adjust_delay_based_on_weather(predicted_delay, sample_data['Simplified_Weather'])

# Further adjust the delay based on airport rating
final_adjusted_delay = adjust_delay_based_on_airport_rating(adjusted_delay, sample_data['Airport Rating'])

if sample_data['Distance'] > 1000:
    final_adjusted_delay -= 5


print(f"Final Adjusted Arrival Delay: {final_adjusted_delay:.2f} minutes")
