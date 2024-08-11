import pandas as pd
from matplotlib import pyplot as plt

from cleandata import data

# Check if 'data' is a DataFrame
if isinstance(data, pd.DataFrame):
    # Check if the required columns exist

    if 'Airline' in data.columns and 'Arrival Delay' in data.columns:
        avg_ARR_delays = data.groupby("Airline")["Arrival Delay"].mean().sort_values()
        print(avg_ARR_delays)
    else:
        print("Error: DataFrame does not contain the required columns.")
else:
    print("Error: 'data' is not a DataFrame.")


plt.bar(avg_ARR_delays.index,avg_ARR_delays)
plt.xlabel('Airline')
plt.ylabel('Mean ARR_delay')
plt.xticks(rotation=90)
plt.show()