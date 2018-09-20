import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_main = pd.read_csv('clinical/clinical_patient_lihc.txt', sep='\t', lineterminator='\r')
df_main.info()

df_drug = pd.read_csv('clinical/clinical_drug_lihc.txt', sep='\t', lineterminator='\r')
df_drug.info()

df_followup = pd.read_csv('clinical/clinical_follow_up_v4.0_lihc.txt', sep='\t', lineterminator='\r')
df_followup.info()

df_followup_nte = pd.read_csv('clinical/clinical_follow_up_v4.0_nte_lihc.txt', sep='\t', lineterminator='\r')
df_followup_nte.info()

df_nte = pd.read_csv('clinical/clinical_nte_lihc.txt', sep='\t', lineterminator='\r')
df_nte.info()

df_omf = pd.read_csv('clinical/clinical_omf_v4.0_lihc.txt', sep='\t', lineterminator='\r')
df_omf.info()

df_radiation = pd.read_csv('clinical/clinical_radiation_lihc.txt', sep='\t', lineterminator='\r')
df_radiation.info()
