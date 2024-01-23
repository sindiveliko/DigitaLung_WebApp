"""
This module contains the introduction information for the homepage - or first page.
"""

import streamlit as st


def display_welcoming_info():
    """
    Displays welcoming information with st.markdown for the user.
    """
    st.markdown("## Welcome to *DigitaLung*")
    st.markdown(
        "Analysing and processing medical time series data is not a trivial task, due to the complexity and "
        "differences in such data. **_DigitaLung_** serves as a tool for healthcare professionals and data science "
        "specialists. It presumes that users have fundamental knowledge, but no in-depth expertise, allowing for "
        "broader accessibility. In the form of a **_web application_**, it allows users to do diverse tasks in "
        "respiratory data analysis, e.g., preprocessing, data visualization, and unsupervised learning mostly "
        "by themselves. Therefore, **_DigitaLung_** does not only provide functions to process timeseries, "
        "but also provides enough background knowledge so that non-professionals are able to use the app with "
        "limited prior experience.")
    st.write()
    st.markdown(
        "On the left side you find the sidebar, divided in three sections: \n"
        "- *Navigation*: Here you can navigate pages and choose between different "
        "tasks. \n"
        "- *File Management*: Buttons can be used when uploaded a wrong dataset, or reverse steps. \n"
        "   - *Reset App* restarts the Session and all data is deleted. \n"
        "   - To delete the current progress but keep the uploaded files and extracted patient information, use "
        "*Delete Progress*. \n"
        "- *File State*: To know more about the current state of your work progress within the webapp, you can "
        "navigate the checkboxes.  \n" 
        "   See uploaded file names and search through them, get information about data entry and extraction, as well "
        "as which preprocessing steps were performed.")
    st.markdown(
        "The audio files (aka your time series) uploaded are expected to have a certain format. They must be "
        "WAV File Format.  \nMoreover, at this stage of development, the webapp is developed and functions based on "
        "the [ICBHI Respiratory Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database).")
    st.write()
    st.markdown("##### Data usage by task")
    st.markdown("*Preprocessing* - uses the *uploaded data* directly")
    st.markdown("*Channel Analysis* - uses the *uploaded data* directly")
    st.markdown("*Audio Scaling* - uses the *uploaded data* directly")
    st.markdown("*Visualization* - uses the *scaled data*, if present. "
                "Otherwise, uses *uploaded data*. ")
    st.markdown("*Extract Respiratory Cycles* - uses the *uploaded data* directly")
    st.markdown("*Imputation* - uses the *respiratory cycles*")
    st.markdown("*Clustering* - uses the *respiratory cycles* or *imputed data*, when present.")
