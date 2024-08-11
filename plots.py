import matplotlib.pyplot as plt

from model import data

# Plot histogram of arrival delay in the dataset
plt.figure(figsize=(10, 6))
plt.hist(data['Arrival Delay'], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Arrival Delay')
plt.ylabel('Frequency')
plt.title('Distribution of Arrival Delay')
plt.show()





