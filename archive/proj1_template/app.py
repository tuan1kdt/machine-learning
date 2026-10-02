import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards

from Statistics import Statistics
from AboutMe import AboutMe
from Home import Home

st.set_page_config(page_title="Medical Cost Prediction", page_icon="🌍", layout="wide")

#all graphs we use custom css not streamlit 
theme_plotly = None 


# load Style css
def load_custom_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Call this function in your `main()` to load the CSS
load_custom_css()


#menu bar
def sideBar():
    with st.sidebar:
       selected=option_menu(
           menu_title="Main Menu",
           options=["Home", "Statistics", "About Me"],
           icons=["house", "eye", "people"],
           menu_icon="cast",
           default_index=0
       )
    if selected=="Home":
        Home()
    if selected=="Statistics":
        Statistics()
    if selected=="About Me":
        AboutMe()

st.sidebar.image("logo.png", caption="")
sideBar()



#theme
hide_st_style="""

<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""
