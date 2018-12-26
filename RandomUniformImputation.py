# PURPOSE:
# Do Uniform distribution imputation on exams results based on references values grouped by age and sex
# extracted from peer-reviewed scientific articles and books.
# It should be used when only age and gender columns are filled

import numpy as np
import pandas as pd


def calculate_value(d, exam, gender, age):
    for ir, rr in d[(d.Gender == gender) & (d.Exam == exam)].iterrows():
        if age >= rr['Min Age'] and age <= rr['Max Age']:
            val = 0
            while True:
                val = np.random.normal(rr['Mean'], rr['Standard Deviation'], 10000)[5000]
                if val > 0 and rr['Mean'] - rr['Standard Deviation'] <= val <= rr['Mean'] + rr['Standard Deviation']:
                    break
            return val


df_rr = pd.read_csv('csv/reference-range.csv')
df = pd.read_csv('csv/input-uniform.csv')

for i, r in df.iterrows():
    df.at[i, 'AFP'] = calculate_value(df_rr, 'AFP', r['Gender'], r['Age'])
    df.at[i, 'Platelets'] = calculate_value(df_rr, 'Platelets', r['Gender'], r['Age'])
    df.at[i, 'Prothrombin Time'] = calculate_value(df_rr, 'Prothrombin Time', r['Gender'], r['Age'])
    df.at[i, 'Albumin'] = calculate_value(df_rr, 'Albumin', r['Gender'], r['Age'])
    df.at[i, 'Total Bilirubin'] = calculate_value(df_rr, 'Total Bilirubin', r['Gender'], r['Age'])
    df.at[i, 'Creatinine'] = calculate_value(df_rr, 'Creatinine', r['Gender'], r['Age'])

df.to_csv('csv/uniform-inputation.csv')
