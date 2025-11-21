import streamlit as st
import os
import numpy as np



# --- Streamlit page setup ---
st.set_page_config(page_title="Machine learning with MLB statistics", page_icon="⚾️", layout="centered")
st.title("⚾️ MLB Pitching")
st.write("""
Predicting Baseball Hall of Fame selections with machine learning.

I analyzed a dataset of 1215 Major League Baseball pitchers and attempted to predict whether each one has been named to the National Baseball Hall of Fame. 

Two criteria have been applied: the pitcher must have thrown a minimum of 1000 career innings (to ensure longevity), and must have retired no later than 2019 (to account for the time delay in eligibility). 
""")

st.title("🤔😵‍💫 Data challenges")
st.write("""
This dataset is highly imbalanced - just 82 of the 1215 pitchers are in the Hall of Fame.

There is some dispute as to what qualifies as a Major League, especially pre-1901. Some accepted modern teams were previously in leagues that have since disbanded, and some legendary players spent part or all of their careers with teams that no longer exist. I used every league that Stathead judges to be major.

Older data are incomplete, especially for the Negro Leagues, where many barnstorming games were played for which there is little or no information available.

Additionally, some players with unimpressive playing careers may later achieve notoriety through coaching or management and be inducted for those achievements.

Most importantly, Hall of Fame induction is not based exclusively on objective statistics, but is voted on by the Baseball Writers' Association of America (BBWAA), whose motivations are often opaque and shift over time. Some players with impressive statistics are excluded for non-statistical reasons, such as admitted or suspected steroid usage. It has also been implied that personality factors and teams played for may have influenced voting decisions.

I wanted to see how the models could handle these quirks.
""")

st.title("✨🧠🖥️ About my models and process")
st.write("""
I used seven classification models: Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Booster, Support Vector, and Neural Network.

For each of my models, I ran K-Fold regularization so that every player would be predicted. This was to have more data points to work with, and to compare predictions for the same player across all models.

For most of the machine learning models, I ran GridSearchCV to find optimal hyperparameters. Since there is no real-world harm to be done by an incorrect prediction, in theory, Precision and Recall should be equally important. For this reason, my GridSearches were originally run on F1 Score. However, finding a good Precision score proved to be the biggest challenge, which is to say that many players were falsely predicted to be Hall of Famers when in fact they are not. Therefore, I also ran the GridSearch on Precision and Recall, each also combined with both standard and stratified K-Folds, to see the tradeoffs.

I did not run a GridSearch on the Logistic Regression due to its simplicity, but I did run a ridge and lasso. Furthermore, the Support Vector Machine did not run a GridSearch in over 30 minutes, so I was forced to suspend it for time purposes, but I was able to run both versions of K-Folds. 
and the Support Vector Machine

I used a Standard Scaler on the Logistic Regression, SVM, and Neural Network, but tree-based models were not scaled, as doing so reduced performance. I also tried using a feature-reduced version of my dataset, but in each model the performance was slightly worse. Furthermore, I also attempted inverting /"lower is better/" features such as ERA, so that a higher number would always be more favorable, but this had almost zero bearing on performance. 

Once all modeling was complete, I saved the results as CSV files and loaded every version into a comparison dataframe, from which I chose the best version of each model.

To see my process in detail, visit:
https://github.com/matthewcavanaugh96/magnimind/tree/main/Capstone
""")

st.title("Try it for yourself!")
st.write("""
Search for a player and see how the models did!
""")

import streamlit as st
import pandas as pd
import unicodedata

# --------------------------------------
# Load your final dataframe
# --------------------------------------
df = pd.read_csv("stored_datasets/All Pitchers Cleaned.csv")

# Function to normalize names for accent-insensitive search
def normalize(s):
    s = unicodedata.normalize('NFD', s)
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn').lower()

# Create normalized lookup column (safe to do here too)
df["normalized"] = df["Player"].apply(normalize)

st.title("MLB Hall of Fame Pitcher Prediction Explorer")

# --------------------------------------
# Search Box (returns all matches)
# --------------------------------------
search_query = st.text_input("Search for a pitcher (partial name ok):", "")

matches = []

if search_query.strip() != "":
    q = normalize(search_query)
    matches = df[df["normalized"].str.contains(q)]["Player"].tolist()

    if len(matches) == 0:
        st.warning("No players found.")
    else:
        selected_player = st.selectbox("Select a player:", matches)
else:
    selected_player = None

# --------------------------------------
# Show player stats only after they pick
# --------------------------------------
if selected_player:
    player_row = df[df["Player"] == selected_player].iloc[0]

    st.subheader(f"📌 {selected_player}")

    # Example metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("HOF Status", player_row["Hall of Fame"])
    col2.metric("Wins", f"{player_row['Wins']:.2f}")
    col3.metric("Earned Run Average", f"{player_row['Earned Run Average']:.2f}")
    col4.metric("Strikeouts", f"{player_row['Strikeouts']:.2f}")


    st.markdown("---")

    # Show stats table
    st.dataframe(player_row.to_frame("Value"))

