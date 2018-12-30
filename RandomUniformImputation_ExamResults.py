# PURPOSE:
# Do Uniform distribution imputation on exams results based on references values grouped by age and sex
# extracted from peer-reviewed scientific articles and books.
# It should be used when only age and gender columns are filled

import numpy as np
import pandas as pd
import math

def truncate(number, digits) -> float:
    stepper = pow(10.0, digits)
    return math.trunc(stepper * number) / stepper

def calculate_value(d, exam, gender, age):
    for ir, rr in d[(d.Gender == gender) & (d.Exam == exam)].iterrows():
        if age >= rr['Min Age'] and age <= rr['Max Age']:
            val = 0
            while True:
                val = np.random.normal(rr['Mean'], rr['Standard Deviation'], 10000)[5000]
                if val > 0 and rr['Mean'] - rr['Standard Deviation'] <= val <= rr['Mean'] + rr['Standard Deviation']:
                    break
            return val


df_rr = pd.read_csv('csv/reference-range-examresults.csv')
df = pd.read_csv('csv/input-uniform-examresults-cptac.csv')

for i, r in df.iterrows():
    df.at[i, 'AFP'] = truncate(calculate_value(df_rr, 'AFP', r['Gender'], r['Age']), 0)
    df.at[i, 'Platelets'] = truncate(calculate_value(df_rr, 'Platelets', r['Gender'], r['Age']), 0)
    df.at[i, 'Prothrombin Time'] = truncate(calculate_value(df_rr, 'Prothrombin Time', r['Gender'], r['Age']), 1)
    df.at[i, 'Albumin'] = truncate(calculate_value(df_rr, 'Albumin', r['Gender'], r['Age']), 1)
    df.at[i, 'Total Bilirubin'] = truncate(calculate_value(df_rr, 'Total Bilirubin', r['Gender'], r['Age']), 1)
    df.at[i, 'Creatinine'] = truncate(calculate_value(df_rr, 'Creatinine', r['Gender'], r['Age']), 1)

df.to_csv('csv/uniform-inputation-examresults-cptac.csv')
