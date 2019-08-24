# PURPOSE:
# Write results in latex tables

import numpy as np
import pandas as pd


def trunc(txt):
    return np.around(txt, decimals=3)


def print_row(df_wo, df_w, txtTitle, txtColumn):
    print(txtTitle, ' & ', trunc(df_wo['Avg '+txtColumn]), ';', trunc(df_w['Avg '+txtColumn]), '&',
          trunc(df_wo['Desvpad '+txtColumn]), ';', trunc(df_w['Desvpad '+txtColumn]), '&',
          trunc(df_wo['Min '+txtColumn]), ';', trunc(df_w['Min '+txtColumn]), '&', trunc(df_wo['Max '+txtColumn]),
          ';', trunc(df_w['Max '+txtColumn]), '&', trunc(df_wo['Median '+txtColumn]), ';',
          trunc(df_w['Median '+txtColumn]), '\\\\')


def select_rows(df, fusion_type, arch_name):
    df_specific = df[(df_r['Fusion'] == fusion_type) & (df_r['Architecture'] == arch_name)]
    df_wo_pre_proc = df_specific.iloc[0]
    df_w_pre_proc = df_specific.iloc[1]

    print_row(df_wo_pre_proc, df_w_pre_proc, 'Acuracia', 'Accuracy')
    print_row(df_wo_pre_proc, df_w_pre_proc, 'Precisao', 'Precision')
    print_row(df_wo_pre_proc, df_w_pre_proc, 'Recall', 'Recall')
    print_row(df_wo_pre_proc, df_w_pre_proc, 'F-Score', 'F-Score')
    print_row(df_wo_pre_proc, df_w_pre_proc, 'Kappa Value', 'Kappa Value')


df_r = pd.read_csv('csv/results.csv')
select_rows(df_r, 'Unimodal', 'Ad Hoc Light')