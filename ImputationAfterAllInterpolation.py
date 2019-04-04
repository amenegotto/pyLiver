import pandas as pd

# PURPOSE:
# After all the data prepare and preprocessing for multimodal training, some patients without data were found.
# Here spline interpolation is used as imputation method to fill this gap

df = pd.read_csv('csv/clinical_data.csv')

df['Gender'] = df['Gender'].interpolate(method='spline', order=1)
df['Age at Diagnosis'] = df['Age at Diagnosis'].interpolate(method='spline', order=1)
df['Height'] = df['Height'].interpolate(method='spline', order=1)
df['Weight'] = df['Weight'].interpolate(method='spline', order=1)
df['Race'] = df['Race'].interpolate(method='spline', order=1)
df['Etnicity'] = df['Etnicity'].interpolate(method='spline', order=1)
df['Other Malignancy'] = df['Other Malignancy'].interpolate(method='spline', order=1)
df['Family History Cancer Indicator'] = df['Family History Cancer Indicator'].interpolate(method='spline', order=1)
df['Family History Cancer  Number Relatives'] = df['Family History Cancer  Number Relatives'].interpolate(method='spline', order=1)
df['Alcohol'] = df['Alcohol'].interpolate(method='spline', order=1)
df['Hemochromatosis'] = df['Hemochromatosis'].interpolate(method='spline', order=1)
df['Hepatitis'] = df['Hepatitis'].interpolate(method='spline', order=1)
df['Non-Alcoholic Fatty Liver Disease'] = df['Non-Alcoholic Fatty Liver Disease'].interpolate(method='spline', order=1)
df['Other'] = df['Other'].interpolate(method='spline', order=1)
df['AFP'] = df['AFP'].interpolate(method='spline', order=1)
df['Platelets'] = df['Platelets'].interpolate(method='spline', order=1)
df['Prothrombin Time'] = df['Prothrombin Time'].interpolate(method='spline', order=1)
df['Albumin'] = df['Albumin'].interpolate(method='spline', order=1)
df['Total Bilirubin'] = df['Total Bilirubin'].interpolate(method='spline', order=1)
df['Creatinine'] = df['Creatinine'].interpolate(method='spline', order=1)

for i, r in df.iterrows():
    df.at[i, 'Gender'] = int(r['Gender'])
    df.at[i, 'Age at Diagnosis'] = round(r['Age at Diagnosis'])
    df.at[i, 'Height'] = round(r['Height'])
    df.at[i, 'Weight'] = round(r['Weight'])
    df.at[i, 'Race'] = int(r['Race'])
    df.at[i, 'Etnicity'] = int(r['Etnicity'])
    df.at[i, 'Other Malignancy'] = round(r['Other Malignancy'])
    df.at[i, 'Family History Cancer Indicator'] = round(r['Family History Cancer Indicator'])
    df.at[i, 'Family History Cancer  Number Relatives'] = round(r['Family History Cancer  Number Relatives'])
    df.at[i, 'Alcohol'] = round(r['Alcohol'])
    df.at[i, 'Hemochromatosis'] = round(r['Hemochromatosis'])
    df.at[i, 'Hepatitis'] = round(r['Hepatitis'])
    df.at[i, 'Non-Alcoholic Fatty Liver Disease'] = round(r['Non-Alcoholic Fatty Liver Disease'])
    df.at[i, 'Other'] = round(r['Other'])
    df.at[i, 'AFP'] = round(r['AFP'])
    df.at[i, 'Platelets'] = round(r['Platelets'])
    df.at[i, 'Prothrombin Time'] = round(r['Prothrombin Time'],1)
    df.at[i, 'Albumin'] = round(r['Albumin'],1)
    df.at[i, 'Total Bilirubin'] = round(r['Total Bilirubin'],1)
    df.at[i, 'Creatinine'] = round(r['Creatinine'],1)


df.to_csv('csv/new_clinical_data.csv')        
