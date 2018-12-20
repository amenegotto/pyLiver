import pandas as pd

# PURPOSE:
# Prepare data for use in random forest regression to do the  missing data imputation of the real clinical data for multimodal machine learning.
# Here spline interpolation is used as imputation method
# The base CSV used here is an ensemble of TCGA-LIHC clinical data and HCC Survival Dataset (UCI)

df = pd.read_csv('csv/hcc-survive.csv')
df['Alpha-Fetoprotein'] = df['Alpha-Fetoprotein'].interpolate(method='spline', order=1)
df['Platelets'] = df['Platelets'].interpolate(method='spline', order=1)
df['Albumin'] = df['Albumin'].interpolate(method='spline', order=1)
df['Total Bilirubin'] = df['Total Bilirubin'].interpolate(method='spline', order=1)
df['Creatinine'] = df['Creatinine'].interpolate(method='spline', order=1)

df.to_csv('csv/hcc-data-filled.csv')        
