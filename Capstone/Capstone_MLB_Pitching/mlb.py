import streamlit as st
import os
import numpy as np



# --- Streamlit page setup ---
st.set_page_config(page_title="MLB Player Batting", page_icon="✨🧠🖥️", layout="centered")
st.title("⚾️ MLB Batting")
st.write("""
Predicting Baseball Hall of Fame selections with machine learning.

About my models:
I used seven classification models:
Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Booster, Support Vector, and Neural Network.
For each of my models, I ran K-Fold regularization so that every player would be predicted. This was to have more data points to work with, and to compare predictions for the same player across all models.

For most of the machine learning models, I ran a GridSearchCV to find optimal parameters for Precision, Recall, and F1 Score. Each of these were also combined with both standard and stratified K-Fold regularization. 

I did not run a GridSearch on the Logistic Regression due to its simplicity, but I did run a ridge and lasso. Furthermore, the Support Vector Machine did not run a GridSearch in over 30 minutes, so I was forced to suspend it for time purposes, but I was able to run both versions of K-Folds. 
and the Support Vector Machine

I used a Standard Scaler on the Logistic Regression, SVM, and Neural Network, but tree-based models were not scaled, as doing so reduced performance. I also tried using a feature-reduced version of my dataset, but in each model the performance was slightly worse. Furthermore, I also attempted inverting /"lower is better/" features such as ERA, so that a higher number would always be more favorable, but this had almost zero bearing on performance. 

Once all modeling was complete, I saved the results as CSV files and loaded every version into a comparison dataframe, from which I chose the best version of each model.

To see my process in detail, visit:
https://github.com/matthewcavanaugh96/magnimind/tree/main/Capstone

""")