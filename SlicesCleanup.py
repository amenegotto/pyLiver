import os
import pandas as pd


# PURPOSE: 
# read a CSV that contains the folder and the initial and final image numbers.
# All slices outside this range is excluded. 

#df = pd.read_csv('csv/cptac.csv')
#df = pd.read_csv('csv/tcga-kirp.csv')
#df = pd.read_csv('csv/tcga-lihc.csv')
df = pd.read_csv('csv/tcga-stad.csv')

for row in df.itertuples():
    print(row)
    for f in os.listdir(row.Directory):
        if not os.path.isdir(row.Directory + f):
            fname = os.path.splitext(f)[0]
            curr = int(fname[1:])
            if (curr < row.Initial or curr > row.Final):
                os.remove(row.Directory + '/' + f)
                # print(row.Directory + f)
