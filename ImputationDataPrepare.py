import pandas as pd

# PURPOSE:
# Prepare data for use in random forest regression to do the  missing data imputation of the real clinical data for multimodal machine learning.
# The base CSV used here is an ensemble of TCGA-LIHC clinical data and HCC Survival Dataset (UCI)

afp_avg = 0
pla_avg = 0
alb_avg = 0
bili_avg = 0
crea_avg = 0

def should_recalculate(num):   
    if pd.isnull(num) or pd.isna(num):
        return True

def recalculate(csv):
    afp_sum = 0
    pla_sum = 0
    alb_sum = 0
    bili_sum = 0
    crea_sum = 0
    total = len(df)

    for index, row in csv.iterrows():
        afp_sum = afp_sum + row['Alpha-Fetoprotein']
        pla_sum = pla_sum + row['Platelets']
        alb_sum = alb_sum + row['Albumin']
        bili_sum = bili_sum + row['Total Bilirubin']
        crea_sum = crea_sum + row['Creatinine']

    afp_avg = afp_sum / total
    pla_avg = pla_sum / total 
    alb_avg = alb_sum / total
    bili_avg = bili_sum / total
    crea_avg = crea_sum / total


df = pd.read_csv('csv/hcc-data.csv')

for i, r in df.iterrows():
    if should_recalculate(r['Alpha-Fetoprotein']):
        recalculate(df)
        df.set_value(i,'Alpha-Fetoprotein', afp_avg)

    if should_recalculate(r['Platelets']):
        recalculate(df)
        df.set_value(i,'Platelets', pla_avg)
    
    if should_recalculate(r['Albumin']):
        recalculate(df)
        df.set_value(i,'Albumin', alb_avg)
        
    if should_recalculate(r['Total Bilirubin']):
        recalculate(df)
        df.set_value(i,'Total Bilirubin', bili_avg)
        
    if should_recalculate(r['Creatinine']):
        recalculate(df)
        df.set_value(i,'Creatinine', crea_avg)

df.to_csv('hcc-data-filled.csv')        
