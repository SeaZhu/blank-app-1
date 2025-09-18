# streamlit_app.py
import streamlit as st

st.set_page_config(page_title="FEVS-style Dashboard", layout="wide")

# Only register the Overview page
overview = st.Page("pages/overview.py", title="Overview", icon="📊")

nav = st.navigation({"Main": [overview]})
nav.run()
