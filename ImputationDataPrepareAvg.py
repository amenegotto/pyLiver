import pandas as pd

# PURPOSE:
# Prepare data for use in random forest regression to do the  missing data imputation of the real clinical data for multimodal machine learning.
# Here average is used as imputation method
# The base CSV used here is an ensemble of TCGA-LIHC clinical data and HCC Survival Dataset (UCI)

afp_avg = 0
pla_avg = 0
alb_avg = 0
bili_avg = 0
crea_avg = 0

def should_recalculate(num):
    if pd.isnull(num) or pd.isna(num):
        return True
    else:
        return False

def recalculate(csv, exam):
    global afp_avg
    global pla_avg
    global alb_avg
    global bili_avg
    global crea_avg
    afp_sum = 0
    pla_sum = 0
    alb_sum = 0
    bili_sum = 0
    crea_sum = 0
    afp_qt = len(df)
    pla_qt = len(df)
    alb_qt = len(df)
    bili_qt = len(df)
    crea_qt = len(df)

    for index, row in csv.iterrows():
        if exam == 'AFP':
            if pd.isna(row['Alpha-Fetoprotein']):
                afp_qt = afp_qt - 1
            else:
                afp_sum = afp_sum + row['Alpha-Fetoprotein']

        if exam == 'PLA':
            if pd.isna(row['Platelets']):
                pla_qt = pla_qt - 1
            else:
                pla_sum = pla_sum + row['Platelets']

        if exam == 'ALB':
            if pd.isna(row['Albumin']):
                alb_qt = alb_qt - 1
            else:
                alb_sum = alb_sum + row['Albumin']

        if exam == 'BIL':
            if pd.isna(row['Total Bilirubin']):
                bili_qt = bili_qt - 1
            else:
                bili_sum = bili_sum + row['Total Bilirubin']

        if exam == 'CREA':
            if pd.isna(row['Creatinine']):
                crea_qt = crea_qt - 1
            else:
                crea_sum = crea_sum + row['Creatinine']

    if exam == 'AFP':
        afp_avg = afp_sum / afp_qt

    if exam == 'PLA':
        pla_avg = pla_sum / pla_qt

    if exam == 'ALB':
        alb_avg = alb_sum / alb_qt

    if exam == 'BIL':
        bili_avg = bili_sum / bili_qt

    if exam == 'CREA':
        crea_avg = crea_sum / crea_qt


df = pd.read_csv('csv/hcc-data.csv')

#recalculate(df)

for i, r in df.iterrows():
    if should_recalculate(r['Alpha-Fetoprotein']):
        recalculate(df, 'AFP')
        df.at[i,'Alpha-Fetoprotein'] = afp_avg

    if should_recalculate(r['Platelets']):
        recalculate(df, 'PLA')
        df.at[i,'Platelets'] =  pla_avg
    
    if should_recalculate(r['Albumin']):
        recalculate(df, 'ALB')
        df.at[i,'Albumin'] = alb_avg
        
    if should_recalculate(r['Total Bilirubin']):
        recalculate(df, 'BIL')
        df.at[i,'Total Bilirubin'] = bili_avg
        
    if should_recalculate(r['Creatinine']):
        recalculate(df, 'CREA')
        df.at[i,'Creatinine'] = crea_avg

df.to_csv('hcc-data-filled.csv')        
