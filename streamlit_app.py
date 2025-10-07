# streamlit_app.py
import streamlit as st

st.set_page_config(page_title="FEVS-style Dashboard", layout="wide")

# Register pages
overview = st.Page("pages/overview.py", title="Overview", icon="📊")
dimensions = st.Page(
    "pages/performance_dimensions.py",
    title="Index Results",
    icon="📈",
)
survey_items = st.Page(
    "pages/survey_item_results.py",
    title="Survey Item Results",
    icon="📝",
)
areas_of_concern = st.Page(
    "pages/areas_of_concern.py",
    title="Areas of Concern",
    icon="🚨",
)
our_strength = st.Page(
    "pages/our_strength.py",
    title="Our Strength",
    icon="💪",
)

nav = st.navigation({
    "Main": [overview, dimensions, survey_items, areas_of_concern, our_strength]
})
nav.run()
