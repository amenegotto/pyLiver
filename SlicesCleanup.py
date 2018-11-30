import os
import pandas as pd


# PURPOSE: 
# read a CSV that contains the folder and the initial and final image numbers.
# All slices outside this range is excluded. 

df = pd.read_csv('csv/slices.csv')

for row in df.itertuples():
    print(row)
    for f in os.listdir(row.folder):
        if not os.path.isdir(row.folder + f):
            fname = os.path.splitext(f)[0]
            curr = int(fname[1:])
            if (curr < row.initial or curr > row.final):
                os.remove(row.folder + f)
