import numpy as np
from matplotlib import pyplot as plt

from cleandata import data

plt.figure(figsize=(20, 5))
plt.hist(data['Departure Delay'], bins=80, label='departure', alpha=.2, color='blue')
plt.hist(data['Arrival Delay'], bins=100, label='arrival', alpha=.2, color='red')
plt.yscale("log")
plt.xticks(np.arange(-180, 2401, 60), labels=np.arange(-180, 2401, 60)//60)
plt.xlim(-180, 2400)
plt.xlabel('Delay in hours', fontsize=18)
plt.ylabel('Counts (log scale)', fontsize=18)
plt.title('Departure/Arrival delays in hours', fontsize=23)
plt.legend()
plt.grid()
plt.show()
