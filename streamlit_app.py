# streamlit_app.py
import streamlit as st

from fevs_style import apply_global_styles

st.set_page_config(page_title="FEVS-style Dashboard", layout="wide")
apply_global_styles()

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
opportunities_to_improve = st.Page(
    "pages/opportunities_to_improve.py",
    title="Opportunities to Improve",
    icon="🧭",
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
distribution = st.Page(
    "pages/distribution_of_survey_items.py",
    title="Distribution of Survey Items",
    icon="📈",
)

nav = st.navigation(
    {
        "Main": [
            overview,
            dimensions,
            survey_items,
            areas_of_concern,
            our_strength,
            opportunities_to_improve,
            distribution,
        ]
    }
)
nav.run()
