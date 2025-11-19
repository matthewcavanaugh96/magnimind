import streamlit as st
import os
import numpy as np



# --- Streamlit page setup ---
st.set_page_config(page_title="MLB Player Batting", page_icon="✨🧠🖥️", layout="centered")
st.title("⚾️ MLB Batting")
st.write("""
Predicting Baseball Hall of Fame selections with machine learning.
""")