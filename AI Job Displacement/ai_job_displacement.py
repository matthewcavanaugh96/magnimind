import streamlit as st


import streamlit as st

# MUST be first Streamlit command
st.set_page_config(
    page_title="Does AI think it's coming for your job?",
    page_icon="🤖",
    layout="centered"
)

# Background and text color
# Inverted, not using this makes things too hard
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f0f2f6;
#         color: #1f2937;
#     }
#     </style>
#     "",
#     unsafe_allow_html=True
# )







st.title("Does AI think it's coming for your job?")
#st.write("Background color example")

# End Config
# ==========
# ==========
# ==========





# Import libraries
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


# --- Streamlit page setup ---


st.title("⚾️ Overview")
st.write("""
Filler text. 

""")


st.title("🖥️ About the project")
st.write("""
More filler text.
""")





# Define datasets
models_df = pd.read_csv('transformed data/3-Model Scores and Consensus.csv')


# df_player_stats = pd.read_csv('final_tables/Player_stats.csv')
# df_player_preds = pd.read_csv('final_tables/Players_and_models.csv')
# df_model_comp = pd.read_csv('final_tables/model_comparison.csv')

col_search, col_select, col_random = st.columns([1.2, 1, 0.6])





# ======


import re
import pandas as pd
import streamlit as st

# -----------------------------
# Helpers
# -----------------------------
def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)   # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()   # collapse spaces
    return s

def build_search_df(models_df: pd.DataFrame) -> pd.DataFrame:
    df = models_df.copy()

    # Expecting an 'aliases' column soon; handle if missing
    if "aliases" not in df.columns:
        df["aliases"] = ""

    # Search corpus = title + aliases
    df["__search_corpus"] = (
        df["Title"].fillna("").astype(str) + " " + df["aliases"].fillna("").astype(str)
    )
    df["__search_norm"] = df["__search_corpus"].map(normalize_text)

    # Friendly label for selection + stable ID
    df["__label"] = df["Title"].astype(str) + "  —  " + df["O*NET-SOC Code"].astype(str)
    df["__job_id"] = df["O*NET-SOC Code"].astype(str) + "||" + df["Title"].astype(str)

    return df

def infer_model_columns(df: pd.DataFrame):
    candidates = [c for c in df.columns if "doubleweighed" in c.lower()]
    return [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]

# -----------------------------
# Session state init
# -----------------------------
if "selected_job_id" not in st.session_state:
    st.session_state.selected_job_id = None

# -----------------------------
# UI
# -----------------------------
st.title("AI Job Displacement Lookup")

df = build_search_df(models_df)
model_cols = infer_model_columns(df)

# Random job button
col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("🎲 Random job", use_container_width=True):
        rnd = df.sample(1, random_state=None).iloc[0]
        st.session_state.selected_job_id = rnd["__job_id"]
with col_b:
    search = st.text_input(
        "Start typing a job title (aliases included)",
        placeholder="e.g., IT, I.T., software dev, registered nurse",
    )

# Filter based on search
if search.strip():
    q = normalize_text(search)
    filtered = df[df["__search_norm"].str.contains(q, na=False)].copy()
else:
    filtered = df.sort_values("Consensus Rank").head(50).copy() if "Consensus Rank" in df.columns else df.head(50).copy()

max_results = st.slider("Max results", 10, 200, 50, step=10)
filtered = filtered.head(max_results)

if filtered.empty:
    st.warning("No matches. Try fewer words, different spelling, or an alias.")
    st.stop()

# Determine default selection index (so random button snaps the dropdown)
job_id_list = filtered["__job_id"].tolist()
default_index = 0
if st.session_state.selected_job_id in job_id_list:
    default_index = job_id_list.index(st.session_state.selected_job_id)

selected_label = st.selectbox(
    "Select a job",
    options=filtered["__label"].tolist(),
    index=default_index,
)

selected_row = filtered.loc[filtered["__label"] == selected_label].iloc[0]
st.session_state.selected_job_id = selected_row["__job_id"]

# -----------------------------
# Display
# -----------------------------
st.subheader(selected_row["Title"])
st.caption(f"O*NET-SOC Code: {selected_row['O*NET-SOC Code']}")

# Quick metrics
c1, c2, c3, c4 = st.columns(4)
if "Multi Model Consensus" in df.columns and pd.notna(selected_row["Multi Model Consensus"]):
    c1.metric("Consensus", f"{selected_row['Multi Model Consensus']:.3f}")
if "Consensus Rank" in df.columns and pd.notna(selected_row["Consensus Rank"]):
    c2.metric("Consensus rank", int(selected_row["Consensus Rank"]))
if "Model Disagreement" in df.columns and pd.notna(selected_row["Model Disagreement"]):
    c3.metric("Disagreement", f"{selected_row['Model Disagreement']:.3f}")
if "Displacement Risk Category v2" in df.columns and pd.notna(selected_row["Displacement Risk Category v2"]):
    c4.metric("Risk category", str(selected_row["Displacement Risk Category v2"]))

st.divider()

st.markdown("### Model predictions")
if model_cols:
    model_table = (
        pd.DataFrame({
            "Model": [c.replace("_doubleweighed", "") for c in model_cols],
            "Score": [selected_row[c] for c in model_cols],
        })
        .sort_values("Score", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(model_table, use_container_width=True)
else:
    st.info("No model columns found (expected columns containing 'doubleweighed').")

with st.expander("Show aliases / search text"):
    st.write("Aliases:", selected_row.get("aliases", ""))
    st.write("Search corpus:", selected_row["__search_corpus"])










# # =====
# # =====
# # =====
# # =====
# # =====

# import re
# import pandas as pd
# import streamlit as st

# # -----------------------------
# # Helpers
# # -----------------------------
# def normalize_title(s: str) -> str:
#     """Lowercase, remove punctuation, collapse whitespace."""
#     if pd.isna(s):
#         return ""
#     s = s.lower()
#     s = re.sub(r"[^a-z0-9\s]", " ", s)   # remove punctuation
#     s = re.sub(r"\s+", " ", s).strip()   # collapse spaces
#     return s

# def build_search_df(models_df: pd.DataFrame) -> pd.DataFrame:
#     df = models_df.copy()
#     df["__title_norm"] = df["Title"].map(normalize_title)
#     # Nice label for dropdown results (include SOC for disambiguation)
#     df["__label"] = df["Title"].astype(str) + "  —  " + df["O*NET-SOC Code"].astype(str)
#     return df

# def infer_model_columns(df: pd.DataFrame):
#     """
#     Auto-detect model score columns so adding new models later is easy.
#     Picks numeric columns that look like model outputs (contains 'doubleweighed' by your naming),
#     but you can broaden this rule if you add other naming schemes.
#     """
#     candidates = [c for c in df.columns if "doubleweighed" in c.lower()]
#     # keep only numeric-ish
#     out = []
#     for c in candidates:
#         if pd.api.types.is_numeric_dtype(df[c]):
#             out.append(c)
#     return out

# # -----------------------------
# # Streamlit UI
# # -----------------------------
# st.title("AI Job Displacement Lookup")

# # models_df is assumed to already exist in your app
# # models_df = ...

# df = build_search_df(models_df)

# model_cols = infer_model_columns(df)

# search = st.text_input("Start typing a job title", placeholder="e.g., data scientist, nurse, electrician")

# # Filter results as user types (fast substring match on normalized title)
# if search.strip():
#     q = normalize_title(search)
#     filtered = df[df["__title_norm"].str.contains(q, na=False)].copy()
# else:
#     # Show a small “starter list” when empty, so the UI isn’t blank
#     filtered = df.sort_values("Consensus Rank").head(50).copy() if "Consensus Rank" in df.columns else df.head(50).copy()

# # Limit list length to keep UI snappy
# max_results = st.slider("Max results", 10, 200, 50, step=10)
# filtered = filtered.head(max_results)

# # If nothing matches, be explicit
# if filtered.empty:
#     st.warning("No matches. Try fewer words or a different spelling.")
#     st.stop()

# # Use a selectbox with friendly labels; return the selected label
# selected_label = st.selectbox(
#     "Select a job",
#     options=filtered["__label"].tolist(),
# )

# # Get selected row
# selected_row = filtered.loc[filtered["__label"] == selected_label].iloc[0]

# # -----------------------------
# # Display
# # -----------------------------
# st.subheader(selected_row["Title"])
# st.caption(f"O*NET-SOC Code: {selected_row['O*NET-SOC Code']}")

# # Quick “headline” metrics
# c1, c2, c3, c4 = st.columns(4)
# if "Multi Model Consensus" in df.columns:
#     c1.metric("Consensus", f"{selected_row['Multi Model Consensus']:.3f}")
# if "Consensus Rank" in df.columns:
#     c2.metric("Consensus rank", int(selected_row["Consensus Rank"]) if pd.notna(selected_row["Consensus Rank"]) else "—")
# if "Model Disagreement" in df.columns:
#     c3.metric("Disagreement", f"{selected_row['Model Disagreement']:.3f}" if pd.notna(selected_row["Model Disagreement"]) else "—")
# if "Displacement Risk Category v2" in df.columns:
#     c4.metric("Risk category", selected_row["Displacement Risk Category v2"])

# st.divider()

# # Model-by-model table
# st.markdown("### Model predictions")
# if model_cols:
#     model_table = (
#         pd.DataFrame({
#             "Model": [c.replace("_doubleweighed", "") for c in model_cols],
#             "Score": [selected_row[c] for c in model_cols],
#         })
#         .sort_values("Score", ascending=False)
#         .reset_index(drop=True)
#     )
#     st.dataframe(model_table, use_container_width=True)
# else:
#     st.info("No model columns found. (Expected columns containing 'doubleweighed'.)")

# # Optional: show the entire row for transparency/debug
# with st.expander("Show all fields for this job"):
#     st.write(selected_row.drop(labels=["__title_norm", "__label"], errors="ignore"))








# # ---------------------------------------------------------
# # 🎲 RANDOM PLAYER BUTTON
# # ---------------------------------------------------------
# with col_random:
#     if st.button("🎲 Random Job"):
#         selected_player = random.choice(models_df["Title"].unique().tolist())


# # =====
# # =====
# # =====
# # =====
# # =====








st.title("Additions and fixes")
st.write("""
1. What other professions might be a good fit for someone who is laid off? Would seek to match jobs with similar tasks, tools, and tech skills, and find a position with somewhat lower risk.
2. Random job button doesn't seem to be working, and search isn't greatly optimized.
""")





# Alt text color example
st.markdown("<h1 style='color:#1f77b4;'>Blue Header</h1>", unsafe_allow_html=True)

st.markdown(
    "<p>This is <span style='color:red;'>red text</span> and this is <span style='color:green;'>green text</span>.</p>",
    unsafe_allow_html=True
)


