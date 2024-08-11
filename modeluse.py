import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from model import X

# Load the trained model
model_filename = 'flight_delay_prediction_model_lightgbm_optimized.pkl'
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
        'Moderate or heavy rain shower': 'Rain'
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

    print(f"Preprocessed Sample Data:\n{df.head()}")
    print(f"Scaled Sample Data:\n{df_scaled[:5]}")

    return df_scaled


# Sample data
sample_data = {
    'From': "BOM", 'To': "DEL",
    'Departure Delay': 56,
    'Airline': "Spice Jet",
    'Simplified_Weather': "Sunny",
    'Distance Category': "501-1000 km",
    'Distance': 669, 'Airline Rating': 0.5,
    'Airport Rating': 0.1
}

# Load the scaler used during training
scaler_filename = 'scaler.pkl'  # Adjust the filename if needed
scaler = joblib.load(scaler_filename)

# Preprocess sample data
sample_df_scaled = preprocess_sample_data(sample_data, scaler=scaler)

# Predict delay using the trained model
predicted_delay = loaded_model.predict(sample_df_scaled)
print(f"Predicted Arrival Delay: {predicted_delay[0]:.2f} minutes")
