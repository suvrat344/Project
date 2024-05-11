# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dispaly a title
st.title("House Price Prediction")

# Load the dataset
df = pd.read_csv("Housing.csv")

# Display a table data
if(st.checkbox("View dataset in table data format")):
    st.dataframe(df)

# Show each column description when checkbox is ON
if(st.checkbox("Show each column name and its description")):
    st.markdown(r'''
                ### Column name and its Description
                #### CRIM : Crime occurence rate per unit population by town
                #### ZN : Percentage of 25000-squared feet-area house
                #### INDUS : Percentage of non-retail land area by town
                #### CHAS : Index for Charlse river:0 is near,1 is far
                #### NOX : Nitrogen compund concentration
                #### RM : Average number of rooms per residence
                #### AGE : Percentage of buildings built before 1940
                #### DIS : Weighted distance from five employment centres
                #### RAD : Index for easy access to highway
                #### TAX : Tax rate per 100,000 dollar
                #### PTRATIO : Percentage of students and teachers in each town
                #### B : 1000(Bk-0.63)^2,where Bk is the percentage of Black people
                #### LSTAT : Percentage of low-class population
                ''') 
    
# Plot the relation between target and explanatory variables
if(st.checkbox("Plot the relation between target and explanatory variables")):
    checked_variable = st.selectbox("Select one explanatory variable",df.drop(columns="price").columns)
    fig,ax = plt.subplots(figsize=(5,3))
    ax.scatter(x=df[checked_variable],y=df["price"])
    plt.xlabel(checked_variable)
    plt.ylabel("Price")
    st.pyplot(fig)
    
