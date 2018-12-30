# PURPOSE:
# Random Weighted imputation for race and etnicity based on probabilities extracted from
# "Overview of Race and Hispanic Origin: 2010" (USA 2010 Census).

import numpy as np

CSV_FILENAME='csv/weighted-random-race_etnicity.csv'

race = np.random.choice([2, 1, 0], 32, p=[0.048, 0.367, 0.585])
etnicity = np.random.choice([1, 0], 32, p=[0.163, 0.837])

with open(CSV_FILENAME, "a") as f:
    for i in range(32):
        f.write(str(race[i]) + ',' + str(etnicity[i]) + '\n')


