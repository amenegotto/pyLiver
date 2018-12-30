# PURPOSE:
# Do Uniform distribution imputation on anthroprometric  measures based on values
# extracted from Anthropometric Reference Data for Children and Adults: United States, 2011–2014.

import numpy as np
import pandas as pd
import math


def pounds_to_kg(val):
    return val * 0.45359237


def inches_to_cm(val):
    return val * 2.54


def truncate(number, digits) -> float:
    stepper = pow(10.0, digits)
    return math.trunc(stepper * number) / stepper


def calculate_height(d, gender, age, race, etnicity):
    for ir, rr in d[(d.Gender == gender)].iterrows():
        if race == rr['Race'] and etnicity == rr['Etnicity'] and age >= rr['Min Age'] and age <= rr['Max Age']:
            val = 0
            while True:
                val = np.random.normal(rr['Height Mean'], rr['Height Std'], 10000)[5000]
                if val > 0 and rr['Height Mean'] - rr['Height Std'] <= val <= rr['Height Mean'] + rr['Height Std']:
                    break
            return val


def calculate_weight(d, gender, age, race, etnicity):
    for ir, rr in d[(d.Gender == gender)].iterrows():
        if race == rr['Race'] and etnicity == rr['Etnicity'] and age >= rr['Min Age'] and age <= rr['Max Age']:
            val = 0
            while True:
                val = np.random.normal(rr['Weight Mean'], rr['Weight Std'], 10000)[5000]
                if val > 0 and rr['Weight Mean'] - rr['Weight Std'] <= val <= rr['Weight Mean'] + rr['Weight Std']:
                    break
            return val


df_rr = pd.read_csv('csv/reference-range-anthroprometric.csv')

# first convert weight (pounds => kg) and height (inches => cm) for reference ranges
for i, r in df_rr.iterrows():
    df_rr.at[i, 'Height Mean'] = inches_to_cm(r['Height Mean'])
    df_rr.at[i, 'Height Std'] = inches_to_cm(r['Height Std'])
    df_rr.at[i, 'Weight Mean'] = pounds_to_kg(r['Weight Mean'])
    df_rr.at[i, 'Weight Std'] = pounds_to_kg(r['Weight Std'])

# next step is the imputation
df = pd.read_csv('csv/input-uniform-anthroprometric.csv')
for i, r in df.iterrows():
    df.at[i, 'Height'] = truncate(calculate_height(df_rr, r['Gender'], r['Age'], r['Race'], r['Etnicity']), 0)
    df.at[i, 'Weight'] = truncate(calculate_weight(df_rr, r['Gender'], r['Age'], r['Race'], r['Etnicity']), 0)

df.to_csv('csv/uniform-inputation-anthroprometric.csv')
