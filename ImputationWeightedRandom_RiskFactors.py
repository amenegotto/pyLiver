# PURPOSE:
# Random Weighted imputation for HCC risk factors based on probabilities extracted from peer-reviewed articles.
import numpy as np

CSV_FILENAME='csv/weighted-random.csv'

alcohol = np.random.choice([1, 0], 109, p=[0.182, 0.818])
hemocromatosis = np.random.choice([1, 0], 109, p=[0.025, 0.975])
hepatitis = np.random.choice([1, 0], 109, p=[0.018, 0.982])
nafld = np.random.choice([1, 0], 109, p=[0.03, 0.97])
other = np.random.choice([1, 0], 109, p=[0.01, 0.99])

with open(CSV_FILENAME, "a") as f:
    for i in range(109):
        f.write(str(alcohol[i]) + ',' + str(hemocromatosis[i]) + ',' + str(hepatitis[i]) + ',' + str(nafld[i]) + ',' + str(other[i]) + '\n')


