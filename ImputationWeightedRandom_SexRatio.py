# PURPOSE:
# Random Weighted imputation for sex-ratio based on probabilities extracted from peer-reviewed articles.
import numpy as np

CSV_FILENAME='csv/weighted-random-sexratio.csv'

gender = np.random.choice([1, 0], 32, p=[0.56, 0.44])

with open(CSV_FILENAME, "a") as f:
    for i in range(32):
        f.write(str(gender[i]) + '\n')


