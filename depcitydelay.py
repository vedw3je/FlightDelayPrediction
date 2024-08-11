import pandas as pd

from cleandata import data


result = (
    data
    .query('`Departure Delay` >= 60')
    .From
    .value_counts(normalize=True)
    .head(10) * 100
)
print(result)
