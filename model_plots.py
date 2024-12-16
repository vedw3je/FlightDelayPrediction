import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Load the trained model and scaler
model_filename = 'flight_delay_prediction_model_lightgbm.pkl'
scaler_filename = 'scaleroptimize.pkl'

model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Load and preprocess the test data
data = pd.read_csv("Cleaned_Dataset.csv")
data.fillna(method='ffill', inplace=True)

data['Departure Delay'] = data['Departure Delay'].replace(-1, np.nan)
data['Arrival Delay'] = data['Arrival Delay'].replace(-1, np.nan)

# Drop rows where 'Arrival Delay' is NaN
data = data.dropna(subset=['Arrival Delay'])
data = data.dropna(subset=['Departure Delay'])

# Define distance bins and labels
distance_bins = [0, 500, 1000, 1500, 2000, 2500, np.inf]
distance_labels = ['0-500 km', '501-1000 km', '1001-1500 km', '1501-2000 km', '2001-2500 km', '2500+ km']

# Create distance categories
data['Distance_Category'] = pd.cut(data['Distance'], bins=distance_bins, labels=distance_labels, right=False)

# Drop the original Distance column
data = data.drop(['Distance'], axis=1)

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
    'Dust': 'Clear',
    'Sand': 'Clear',
    'Smoke': 'Cloudy',
    'Volcanic ash': 'Storm',
    'Squalls': 'Storm',
    'Gusts': 'Storm',
    'High winds': 'Storm',
}

data['Simplified_Weather'] = data['weather__hourly__weatherDesc__value'].map(simplified_weather)

# Drop unnecessary columns
data.drop(columns='weather__hourly__weatherDesc__value', inplace=True)

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=['Airline', 'Simplified_Weather', 'Distance_Category', 'From', 'To'])

# Prepare features and target variable
X = data.drop(['Arrival Delay'], axis=1)
X = X.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
y = data['Arrival Delay']

# Scale features
X_scaled = scaler.transform(X)

# Make predictions
y_pred = model.predict(X_scaled)

# Evaluate the model using predictions
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error with LightGBM: {mse}")
print(f"R^2 Score with LightGBM: {r2}")

# Plot performance metrics
metrics = {'Metric': ['Mean Squared Error', 'R^2 Score'], 'Value': [mse, r2]}
metrics_df = pd.DataFrame(metrics)

# Create a bar plot for the performance metrics
plt.figure(figsize=(8, 5))
sns.barplot(x='Metric', y='Value', data=metrics_df)
plt.title('Model Performance Metrics')
plt.ylabel('Value')
plt.xlabel('Metrics')
plt.ylim(0, max(metrics_df['Value']) + 5)  # Adjust y-axis limit for better visualization
plt.axhline(0, color='grey', lw=0.8, ls='--')  # Add a line at y=0 for reference
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
