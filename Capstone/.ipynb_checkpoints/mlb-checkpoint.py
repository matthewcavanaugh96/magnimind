import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
import pandas as pd
import unicodedata
import streamlit as st
from PIL import Image
import random

# Define datasets
df_player_stats = pd.read_csv('final_tables/Player_stats.csv')
df_player_preds = pd.read_csv('final_tables/Players_and_models.csv')
df_model_comp = pd.read_csv('final_tables/model_comparison.csv')


# --- Streamlit page setup ---
st.set_page_config(page_title="Machine learning with MLB statistics", page_icon="⚾️", layout="centered")

img_sabathia = Image.open("saved pics/cc_sabathia.jpeg")
st.image(img_sabathia, caption='CC Sabathia, inducted to the Baseball Hall of Fame in 2025, his first year of eligibility.', use_container_width=True)

st.title("⚾️ MLB statistics and machine learning")
st.write("""
I analyzed a dataset of 1215 Major League Baseball pitchers and attempted to predict whether each one has been named to the National Baseball Hall of Fame. 

Two criteria have been applied: the pitcher must have thrown a minimum of 1000 career innings (to ensure longevity), and must have retired no later than 2019 (to account for the time delay in eligibility). 
""")

import streamlit as st
from PIL import Image

st.title("🖥️ About the source")
image = Image.open("saved pics/stathead-br.jpg")
st.image(image, use_container_width=True)
st.write("""
Stathead Baseball contains detailed statistics on all levels of professional and college baseball stretching as far back as 1871, including full play-by-play data on all MLB games since 1952. Their parent company Sports Reference also operates Basketball Reference, Pro Football Reference, and many others. Stathead helpfully separates batting from pitching statistics, preventing players from being penalized by occasional, likely poor, appearances in roles opposite their normal one. Their built-in filters allow for a multitude of insights, and I was able to download my dataset easily with little cleaning required.
""")

st.title("😵‍💫 Data challenges")
st.write("""
This dataset is highly imbalanced - just 82 of the 1215 pitchers are in the Hall of Fame.

There is some dispute as to what qualifies as a Major League, especially pre-1901. Some accepted modern teams were previously in leagues that have since disbanded, and some legendary players spent part or all of their careers with teams that no longer exist. I used every league that Stathead judges to be major.

Older data are incomplete, especially for the Negro Leagues, where many barnstorming games were played for which there is little or no information available.

Additionally, some players with unimpressive playing careers may later achieve notoriety through coaching or management and be inducted for those achievements.

Most importantly, Hall of Fame induction is not based exclusively on objective statistics, but is voted on by the Baseball Writers' Association of America (BBWAA), whose motivations are often opaque and shift over time. Some players with impressive statistics are excluded for non-statistical reasons, such as admitted or suspected steroid usage. It has also been implied that personality factors and teams played for may have influenced voting decisions.

I wanted to see how the models could handle these quirks.
""")

st.title("✨🧠 About my models and process")
st.write("""
I used seven classification models: Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Booster, XGBoost, and Support Vector Classifier.

For each of my models, I ran K-Fold regularization so that every player would be predicted. This was to have more data points to work with, and to compare predictions for the same player across all models.

For most of the machine learning models, I ran GridSearchCV to find optimal hyperparameters. Since there is no real-world harm to be done by an incorrect prediction, in theory, Precision and Recall should be equally important. For this reason, my GridSearches were originally run on F1 Score. However, Precision and Recall were prone to extremely poor scores, which is to say the models registered a large number of False Positives or False Negatives, respectively. Therefore, I also ran the GridSearch on these metrics, each also combined with both standard and stratified K-Folds, to see the tradeoffs.

I did not run a GridSearch on the Logistic Regression due to its simplicity. Furthermore, the Support Vector Machine did not run a GridSearch in over 30 minutes, so I was forced to suspend it for time purposes, but I was able to run both versions of K-Folds. 

I used a Standard Scaler on the Logistic Regression and SVM, but tree-based models were not scaled, as doing so reduced performance. I also tried using a feature-reduced version of my dataset, but in each model the performance was slightly worse. Furthermore, I also attempted inverting "lower is better" features such as ERA, so that a higher number would always be more favorable, but this had almost zero bearing on performance. 

Once all modeling was complete, I saved the results as CSV files and loaded every version into a comparison dataframe, from which I chose the best version of each model.

To see my process in detail, visit:
https://github.com/matthewcavanaugh96/magnimind/tree/main/Capstone
""")

st.title("📊🤼‍♂️📊 Model comparison")
st.write("""
In addition to the standard metrics provided by Scikit-learn, I also manually calculated what I'm calling Adjusted Prediction Score (APS). I found this by using predict_proba to pull the confidence in the prediction, and compared it with the actual result. The more confident an incorrect prediction, the lower the APS. Here is how each model did on both the default metrics and mine. Listed with the model names are the GridSearch parameters that produced the best overall metrics.
""")

#st.markdown("---")
#st.subheader("📊 Model Comparison Metrics")

# Dropdown for models
model_list = df_model_comp["Model"].unique().tolist()
selected_model = st.selectbox("Select a model to view its metrics:", model_list)

# Pull row for the selected model
model_row = df_model_comp[df_model_comp["Model"] == selected_model].iloc[0]

# Display as a clean table
metrics_df = model_row.to_frame("Value")
st.dataframe(metrics_df)




# -------
# -------
# -------
# -------

st.title("🫵Pick a player and try it for yourself!")
st.write("""
Search for a player and see how the models did with them! Keep in mind, a player won't appear in this dataset if they don't have 1000 innings pitched or were not retired by 2019.
""")

# ---------------------------------------------------------
# Normalize names for accent-insensitive search
# ---------------------------------------------------------
def normalize(s):
    s = unicodedata.normalize('NFD', s)
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn').lower()

df_player_preds["normalized"] = df_player_preds["Player"].apply(normalize)
df_player_stats["normalized"] = df_player_stats["Player"].apply(normalize)

# ---------------------------------------------------------
# Search + dropdown on ONE BAR
# ---------------------------------------------------------
col_search, col_select, col_random = st.columns([1.2, 1, 0.6])

with col_search:
    search_query = st.text_input("Search for a pitcher:", "", key="pitcher_search")

# Process matches
if search_query.strip():
    q = normalize(search_query)
    matches = df_player_preds[df_player_preds["normalized"].str.contains(q, na=False)]
else:
    matches = pd.DataFrame()

# If no matches
if search_query.strip() and matches.empty:
    st.warning("No players found.")
    selected_player = None
else:
    # Only show dropdown when matches exist
    with col_select:
        if matches.empty:
            selected_player = None
        else:
            # Selectbox with empty default so nothing is auto-selected
            options = ["-- Select --"] + matches["Player"].unique().tolist()
            selected = st.selectbox("Matches:", options)

            selected_player = None if selected == "-- Select --" else selected

# ---------------------------------------------------------
# 🎲 RANDOM PLAYER BUTTON
# ---------------------------------------------------------
with col_random:
    if st.button("🎲 Random Player"):
        selected_player = random.choice(df_player_preds["Player"].unique().tolist())

# -------
# -------
# -------
# -------



# ---------------------------------------------------------
# Display Data
# ---------------------------------------------------------
if selected_player:

    # Pull rows from both dataframes
    pred_row = df_player_preds[df_player_preds["Player"] == selected_player].iloc[0]
    stats_row = df_player_stats[df_player_stats["Player"] == selected_player].iloc[0]

    # =====================================================
    # SUMMARY METRICS (Model Predictions + Actual HOF)
    # =====================================================
    st.subheader(f"📌 Summary for {selected_player} : HOF Status and Predictions")

    # True Hall of Fame value
    true_hof = pred_row["Hall of Fame"]
    true_hof_display = "Yes" if true_hof == 1 else "No"

    # List of model prediction columns
    model_cols = ["Logreg Pred", "DTree Pred", "RFC Pred", "ETC Pred", "GBC Pred", "SVC Pred", "XGB Pred"]

    # Convert to Yes/No for display
    model_results = {
        col: ("Yes" if pred_row[col] == 1 else "No")
        for col in model_cols
    }

    # Create columns for the metrics
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

    col1.metric("Actual HOF?", true_hof_display)
    col2.metric("LogReg", model_results["Logreg Pred"])
    col3.metric("Decision Tree", model_results["DTree Pred"])
    col4.metric("Random Forest", model_results["RFC Pred"])
    col5.metric("Extra Trees", model_results["ETC Pred"])
    col6.metric("Gradient Boosting", model_results["GBC Pred"])
    col7.metric("SVC", model_results["SVC Pred"])
    col8.metric("XGBoost", model_results["XGB Pred"])

    
    # Second row: Mean APS alone
    st.markdown("")  # spacing
    col8 = st.columns(1)[0]
    col8.metric("Mean Adjusted Prediction Score (Highest possible 1, lowest possible -1)", f"{pred_row['Mean Adjusted Prediction Score']:.3f}")


    # ---------------------------------------------------------
    # Blurb based on number of correct models
    # ---------------------------------------------------------

    # Count how many models predicted correctly
    correct_count = 0
    for col in model_cols:
        if pred_row[col] == true_hof:
            correct_count += 1

    # Blurb dictionary
    blurbs = {
        7: "🚀 Right on the money! All models correctly predicted this player.",
        6: "🤏 So close to perfection! We only had one straggler.",
        5: "✅ The consensus was correct, but there was a vocal minority of two.",
        4: "🤷 Seems like the models weren’t sure what to do here. Four got it right, three got it wrong. How to break this tie? Rock paper scissors?",
        3: "🤷 Seems like the models weren’t sure what to do here. Three got it right, four got it wrong. How to break this tie? First one to AGI?",
        2: "😵‍💫 Most of the models were wrong - only two were smart (dumb?) enough to go against the grain. Or is it the HOF voters’ fault?",
        1: "🤯 Only one model got this right! Bragging rights for days!",
        0: "❌❌❌❌❌❌❌ The models were unanimously wrong about this player! What happened here? I blame the BBWAA."
    }

    # Display blurb
#    st.markdown("---")
#    st.subheader("📣 Model Consensus Summary")
    st.write(blurbs[correct_count])


# These may be redundant.
blurb7 = "🚀 Right on the money! All models correctly predicted this player."
blurb6 = "🤏 So close to perfection! We only had one straggler."
blurb5 = "✅ The consensus was correct, but there was a vocal minority of two."
blurb4 = "🤷 Seems like the models weren’t sure what to do here. Four got it right, three got it wrong. How to break this tie? Rock paper scissors?"
blurb3 = "🤷 Seems like the models weren’t sure what to do here. Three got it right, four got it wrong. How to break this tie? First one to AGI?"
blurb2 = "😵‍💫 Most of the models were wrong - only two were smart (dumb?) enough to go against the grain. Or is it the HOF voters’ fault?"
blurb1 = "🤯 Only one model got this right! Bragging rights for days!"
blurb0 = "❌❌❌❌❌❌❌ The models were unanimously wrong about this player! What happened here? I blame the BBWAA."


# -------------
# THIS SECTION MAY BE REDUNDANT since I've added mean APS above
#    st.markdown("---")

    # =====================================================
    # MODEL PREDICTIONS TABLE (from df_player_preds)
    # =====================================================
#    st.subheader("🔮 Model Predictions (Probabilities)")

#    prediction_cols = [c for c in df_player_preds.columns if c.startswith("Pred_")]
#    prediction_df = pred_row[prediction_cols].to_frame("Predicted Probability")

#    st.dataframe(prediction_df)

#    st.markdown("---")

    # =====================================================
    # FULL STATS TABLE (from df_player_stats)
    # =====================================================
#    st.subheader("📘 Full Career Statistics")

#    exclude_cols = ["Player", "normalized"]
#    full_stats_df = stats_row.drop(labels=exclude_cols).to_frame("Value")

#    st.dataframe(full_stats_df)
# ------------


st.title("🕥 What I hope to add later")
st.write("""
I worked with a primitive Neural Network which I intended to incorporate alongside the machine learning models. You can still see it in the model breakdown, but it is excluded from most aggregates, as I have not been able to approximate prediction accuracy in the same manner as the other models. A possible workaround would be to scale confidence for each model, but this may distort insights about the models themselves. If I can surmount this issue, I'd like to experiment with multiple neural networks, with different activation functions, numbers of layers and neurons, and so on.

I'd also like to be able to add images for each player.

I am curious about what would happen if I re-ran this dataset while dropping the players that confused the model the most (i.e. consistenly predicted incorrect with high degrees of confidence). Would the models improve and would confidence improve for the other players?
""")