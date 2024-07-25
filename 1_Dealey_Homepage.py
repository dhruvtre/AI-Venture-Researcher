from Dealey_Shared_Functions import *
from section_dictionary import REPORT_SECTIONS
import streamlit as st

st.set_page_config(page_title="Dealey", page_icon="🏠")

with st.sidebar:
    st.image("Group 5-2.png", width=100)

# Custom CSS for accent color
st.markdown("""
<style>
    .accent-text { color: #ee6e73; }
    .section-header { color: #ee6e73; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("Dealey: AI-Powered Investment Research")
st.write("Empowering VC and PE professionals with rapid, comprehensive insights.")

# Main features in containers
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown('<p class="section-header">Company Profiles</p>', unsafe_allow_html=True)
        st.write("Quick, detailed company snapshots")
        for section in REPORT_SECTIONS["company_profile"]:
            st.markdown(f"✓ {section['name']}")
        st.caption("*Ideal for: Due diligence, competitor analysis*")

with col2:
    with st.container():
        st.markdown('<p class="section-header">Industry Reports</p>', unsafe_allow_html=True)
        st.write("In-depth sector analysis")
        for section in REPORT_SECTIONS["macro_industry_report"]:
            st.markdown(f"✓ {section['name']}")
        st.caption("*Ideal for: Market research, investment thesis*")

# How it works and Toolkit in columns
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-header">How Dealey Works</p>', unsafe_allow_html=True)
    steps = ["1. Choose report type", "2. Enter company/industry name", "3. AI analyzes data", "4. Receive comprehensive report"]
    for step in steps:
        st.markdown(f'<span class="accent-text">▸</span> {step}', unsafe_allow_html=True)

with col2:
    st.markdown('<p class="section-header">Our Toolkit</p>', unsafe_allow_html=True)
    
    # Create a 2x2 grid for toolkit images
    toolkit_col1, toolkit_col2, toolkit_col3, toolkit_col4 = st.columns(4)
    
    # Row 1
    with toolkit_col1:
        st.image("google_logo.png", width=30)
    with toolkit_col2:
        st.image("openai_logo.png", width=30)
    
    # Row 2
    with toolkit_col1:
        st.image("llamaindex_logo.jpeg", width=30)
    with toolkit_col2:
        st.image("streamlit_logo.png", width=30)

# Call to action
st.markdown("---")
st.markdown('<p class="accent-text">Ready to streamline your investment research?</p>', unsafe_allow_html=True)

# Footer
st.caption("For support: dhruv@dealey.in")
