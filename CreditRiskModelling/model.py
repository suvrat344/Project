# Importing Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
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
# Feature Selection Done For Categorical And Numerical Features


# Listing All The Final Feature
features = columns_to_be_kept_numerical + ["MARITALSTATUS","EDUCATION","GENDER","last_prod_enq2","first_prod_enq2"]
df = df[features + ["Approved_Flag"]]


# Label Encoding For The Categorical Feature
print(df["MARITALSTATUS"].unique())
print(df["EDUCATION"].unique())
print(df["GENDER"].unique())
print(df["last_prod_enq2"].unique())
print(df["first_prod_enq2"].unique())

# Ordinal Feature For Education
## SSC : 1
## 12th : 2
## GRADUATE : 3
## UNDER GRADUATE : 3
## POST-GRADUATE : 4
## OTHERS : 1           # Verified By Business And User
## PROFESSIONAL : 3

df.loc[df["EDUCATION"]=="SSC","EDUCATION"] = 1
df.loc[df["EDUCATION"]=="12TH","EDUCATION"] = 2
df.loc[df["EDUCATION"]=="Graduate","EDUCATION"] = 3
df.loc[df["EDUCATION"]=="UNDER GRADUATE","EDUCATION"] = 3
df.loc[df["EDUCATION"]=="POST-GRADUATE","EDUCATION"] = 4
df.loc[df["EDUCATION"]=="OTHERS","EDUCATION"] = 1
df.loc[df["EDUCATION"]=="PROFESSIONAL","EDUCATION"] = 3

df["EDUCATION"] = df["EDUCATION"].value_counts().astype(int)
print(df.info())
df["EDUCATION"].unique()


# label Encoding For Categorical Feature
df_encoded = pd.get_dummies(df,columns = ["MARITALSTATUS","GENDER","last_prod_enq2","first_prod_enq2"])
df_encoded.info()


# Split Data Into Training And Testing Data
y = df_encoded["Approved_Flag"]
x = df_encoded.drop(columns=["Approved_Flag"],axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# Machine Learning Model Fitting

## 1. Random Forest
print("Random Forest Classifier")
rf_classifier = RandomForestClassifier(n_estimators=200,random_state=42)
rf_classifier.fit(x_train,y_train)
y_pred = rf_classifier.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy_score : {accuracy:.2f}")

precision,recall,f1_score,_ = precision_recall_fscore_support(y_test,y_pred)

for i,v in enumerate(["P1","P2","P3","P4"]):
    print(f"Class :{v}")
    print(f"Precision : {precision[i]}")
    print(f"Recall : {recall[i]}")
    print(f"f1_score : {f1_score[i]}")
    
    
# 2. Xg Boost Classifier
print("XgBoostClassifier")
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax',num_class=4)

y = df_encoded["Approved_Flag"]
x = df_encoded.drop(["Approved_Flag"],axis = 1)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=0.2,random_state=42)
xgb_classifier.fit(x_train,y_train)
y_pred = xgb_classifier.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy_score : {accuracy:.2f}")

precision,recall,f1_score,_ = precision_recall_fscore_support(y_test,y_pred)

for i,v in enumerate(["P1","P2","P3","P4"]):
    print(f"Class :{v}")
    print(f"Precision : {precision[i]}")
    print(f"Recall : {recall[i]}")
    print(f"f1_score : {f1_score[i]}")
    
    
# 3. Decision Tree
print("DecisionTreeClassifier")
dt_classifier = DecisionTreeClassifier(max_depth=20,min_samples_split=10)
dt_classifier.fit(x_train,y_train)
y_pred = dt_classifier.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy_score : {accuracy:.2f}")

precision,recall,f1_score,_ = precision_recall_fscore_support(y_test,y_pred)

for i,v in enumerate(["P1","P2","P3","P4"]):
    print(f"Class :{v}")
    print(f"Precision : {precision[i]}")
    print(f"Recall : {recall[i]}")
    print(f"f1_score : {f1_score[i]}")
    
# xgboost is giving me best results.


# Apply Standard Scaler

columns_to_be_scaled = ["Age_Oldest_TL","Age_Newest_TL","time_since_recent_payment","max_recent_level_of_deliq",
                        "recent_level_of_deliq","NETMONTHLYINCOME","Time_With_Curr_Empr"]
for i in columns_to_be_scaled:
    column_data = df_encoded[i].values.reshape(-1,1)
    scaler = StandardScaler()
    scaled_column = scaler.fit_transform(column_data)
    df_encoded[i] = scaled_column
    
    
# Xg Boost Classifier
print("XgBoost Tree Classifier After Standardisation")
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax',num_class=4)

y = df_encoded["Approved_Flag"]
x = df_encoded.drop(["Approved_Flag"],axis = 1)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=0.2,random_state=42)
xgb_classifier.fit(x_train,y_train)
y_pred = xgb_classifier.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy_score : {accuracy:.2f}")

precision,recall,f1_score,_ = precision_recall_fscore_support(y_test,y_pred)

for i,v in enumerate(["P1","P2","P3","P4"]):
    print(f"Class :{v}")
    print(f"Precision : {precision[i]}")
    print(f"Recall : {recall[i]}")
    print(f"f1_score : {f1_score[i]}")
    
    
# Hyperparameter Tuning In XgBoost
print("Xg Boost Tree Classifier After Hyperparameter Tuning")
x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=0.2,random_state=42)

xgb_model = xgb.XGBClassifier(objective="multi:softmax",num_class=4)

param_grid = {
    'n_estimators' : [50,100,200],
    'max_depth' : [3,5,7],
    'learning_rate' : [0.01,0.1,0,2]
}

grid_search = GridSearchCV(estimator = xgb_model,param_grid=param_grid,cv=3,scoring="accuracy",n_jobs = -1)
grid_search.fit(x_train,y_train)
print("Best Hyperparameter : ",grid_search.best_params_)


# Evaluate The Best Model With The Best Hyperparameters On Test Data
best_model = grid_search.best_estimator_
accuracy = best_model.score(x_test,y_test)
print("Test Accuray : ",accuracy)


# Hyperparameter Tuning For XgBoost

param_grid = {
    'n_estimators' : [10,50,100],
    'max_depth' : [3,5,8,10],
    'learning_rate' : [0.001,0.01,0.1,1],
    'alpha' : [1,10,100],
    'colsample_bytree' : [0.1,0.3,0.5,0.7,0.9]
}

answers_grid = {
    "combination" : [],
    "train_Accuracy" : [],
    "test_Accuracy" : [],
    "colsample_bytree" : [],
    "learning_rate" : [],
    "max_depth" : [],
    "alpha" : [],
    "n_estimators" : []
}

# Loop Through Each Combination Of HyperParameters

index = 0

for colsample_bytree in param_grid["colsample_bytree"]:
    for learning_rate in param_grid["learning_rate"]:
        for max_depth in param_grid["max_depth"]:
            for alpha in param_grid["alpha"]:
                for n_estimators in param_grid["n_estimators"]:
                    index = index + 1
                    
                     # Define and Train XGBoost Model
                    model = xgb.XGBClassifier(objective="multi:softmax",num_class=4,colsample_bytree=colsample_bytree,
                                             learning_rate=learning_rate,max_depth=max_depth,alpha=alpha,n_estimators=n_estimators)


                    y = df_encoded["Approved_Flag"]
                    x = df_encoded.drop(["Approved_Flag"],axis=1)

                    label_encoder = LabelEncoder()
                    y_encoded = label_encoder.fit_transform(y)

                    x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=0.2,random_state=42)
                    model.fit(x_train,y_train)
                    y_pred_train = model.predict(x_train)
                    y_pred_test = model.predict(x_test)

                    train_accuracy = accuracy_score(y_train,y_pred_train)
                    test_accuracy = accuracy_score(y_test,y_pred_test)
                    
                    # Include Into The List
                    answers_grid["combination"].append(index)
                    answers_grid["train_Accuracy"].append(train_accuracy)
                    answers_grid["test_Accuracy"].append(test_accuracy)
                    answers_grid["colsample_bytree"].append(colsample_bytree)
                    answers_grid["learning_rate"].append(learning_rate)
                    answers_grid["max_depth"].append(max_depth)
                    answers_grid["alpha"].append(alpha)
                    answers_grid["n_estimators"].append(n_estimators)
                    
                    # Print Result For This Combination
                    print(f"Combination{index}")
                    print(f"colsample_bytree : {colsample_bytree},learning_rate:{learning_rate},max_depth:{max_depth},\
                          alpha:{alpha},n_estimators:{n_estimators}")
                    print(f"Train Accuracy : {train_accuracy:0.2f}")
                    print(f"Test Accuracy : {test_accuracy:0.2f}")
                    print("-"*30)