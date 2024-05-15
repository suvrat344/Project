# Importing Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,precision_recall_fscore_support
import warnings
import os


# Importing The Dataset
a1 = pd.read_excel("CustomerBankData.xlsx")
a2 = pd.read_excel("CIBILData.xlsx")

df1 = a1.copy()
df2 = a2.copy()


# Remove Nulls
df1 = df1.loc[df1["Age_Oldest_TL"] != -99999]

columns_to_be_removed = []
for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)
df2.drop(columns_to_be_removed,axis=1,inplace=True)


# Checking Common Column Name
for i in list(df1.columns):
    if(i in list(df2.columns)):
        print(i)
        

# Merge Two DataFrame
df = pd.merge(df1,df2,how='inner',left_on=["PROSPECTID"],right_on=["PROSPECTID"])


# Check How Many Columns Are Categorical
for i in df.columns:
    if(df[i].dtype=="object"):
        print(i)


# Chi-Square Test
for i in ["MARITALSTATUS","EDUCATION","GENDER","last_prod_enq2","first_prod_enq2","Approved_Flag"]:
    chi2,pval,_,_ = chi2_contingency(pd.crosstab(df[i],df["Approved_Flag"]))
    print(i,"---",pval)
# Since all the categorical features have pval <= 0.5,we will accept all


# Check How Many Are Numerical Column
numeric_columns = []
for i in df.columns:
    if df[i].dtype !="object" and i not in ["Approved_Flag","PROSPECTID"]:
        numeric_columns.append(i)
        

# VIF For Numerical Column
vif_data = df[numeric_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0

for i in range(0,total_columns):
    vif_value = variance_inflation_factor(vif_data,column_index)   
    print(column_index,"---",vif_value)    
    
    if(vif_value <= 6):
        columns_to_be_kept.append(numeric_columns[i])
        column_index = column_index + 1 
    else:
        vif_data = vif_data.drop([numeric_columns[i]],axis=1)
        
   
# Check ANOVA For Columns To Be Kept
columns_to_be_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])
    b = list(df["Approved_Flag"])     
    
    group_P1 = [value for value,group in zip(a,b) if group=="P1"]
    group_P2 = [value for value,group in zip(a,b) if group=="P2"]
    group_P3 = [value for value,group in zip(a,b) if group=="P3"]
    group_P4 = [value for value,group in zip(a,b) if group=="P4"]
    
    f_statistic,p_value = f_oneway(group_P1,group_P2,group_P3,group_P4)
    
    if(p_value <= 0.05):
        columns_to_be_kept_numerical.append(i)