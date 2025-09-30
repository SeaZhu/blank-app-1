# streamlit_app.py
import streamlit as st

st.set_page_config(page_title="FEVS-style Dashboard", layout="wide")

# Register pages
overview = st.Page("pages/overview.py", title="Overview", icon="📊")
dimensions = st.Page(
    "pages/performance_dimensions.py",
    title="Performance Dimensions",
    icon="📈",
)
strength = st.Page(
    "pages/strength_areas.py",
    title="Strength Areas",
    icon="💪",
)

nav = st.navigation({"Main": [overview, dimensions, strength]})
nav.run()
