# PURPOSE:
# Random Weighted imputation for clinical features based on probabilities extracted from peer-reviewed articles.
import numpy as np

CSV_FILENAME='csv/weighted-random.csv'

alcohol = np.random.choice([1, 0], 109, p=[0.1, 0.9])
hemocromatosis = np.random.choice([1, 0], 109, p=[0.02, 0.98])
hepatitis = np.random.choice([1, 0], 109, p=[0.05, 0.95])
nafld = np.random.choice([1, 0], 109, p=[0.08, 0.92])
other = np.random.choice([1, 0], 109, p=[0.04, 0.96])

with open(CSV_FILENAME, "a") as f:
    for i in range(109):
        f.write(str(alcohol[i]) + ',' + str(hemocromatosis[i]) + ',' + str(hepatitis[i]) + ',' + str(nafld[i]) + ',' + str(other[i]) + '\n')


