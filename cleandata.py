import pandas as pd

# Load the dataset
data = pd.read_csv("Dataset.csv")

# Drop unnecessary columns
columns_to_drop = [
    'Used Date', 'Scheduled Departure', 'SDEP', 'Departure', 'DEP',
    'Scheduled Arrival', 'SARR', 'Arrival', 'ARR',
    'Status', 'Passenger Load Factor',
    'Market Share', 'OTP Index', 'weather__hourly__windspeedKmph',
     'weather__hourly__precipMM',
    'weather__hourly__humidity', 'weather__hourly__visibility',
    'weather__hourly__pressure', 'weather__hourly__cloudcover', 'Category'
]

# Remove the columns
data_cleaned = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# Save the cleaned dataset to a new CSV file
cleaned_filename = "Cleaned_Dataset.csv"
data_cleaned.to_csv(cleaned_filename, index=False)

print(f"Cleaned dataset saved as {cleaned_filename}")
