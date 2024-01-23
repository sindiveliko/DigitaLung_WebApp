"""DigitaLung

Entry point for DigitaLung. Coordinates all other Modules and functionality.

This module contains the following functions:
    * main - Entry Point to the app. Starts session-states and includes all the possible tabs.
"""
import os
import random

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from clustering import subset_clustering, clustering
from data_entry_and_extraction import data_extraction
from extract_respiratory_cycles import extract_rc
from file_upload import wav_file_upload
from general_utilities import utilities
from imputations import imputation
from intro_page import intro
from preprocess import preprocess
from visualization import visualization_time_domain
from visualization import visualization_timefreq_domain

from config import seed

np.random.seed(seed)
random.seed(seed)


def main():
    """DigitaLung Entry Point

    Coordinates all interactions between tabs and is the place where all critical information is stored
    in session-states.
    """
    # ############################### GENERAL SETTINGS ################################
    st.set_page_config(layout='wide', page_title="DigitaLung")  # uses the entire screen.

    if "wav_uploader" not in st.session_state:
        st.session_state.wav_uploader = []

    if 'wav_file_names' not in st.session_state:
        st.session_state.wav_file_names = []
        st.session_state.wav_names_dict = {}
        st.session_state.wav_dict_index = 0
        st.session_state.wav_index_dict = {}
        st.session_state.wav_parameters = []
        st.session_state.librosa_sr = 16000     # Sample rate by upload
        st.session_state.wav_dataframes = []    # Uploaded audio material

        # Extracted Information
        st.session_state.wav_rec_dict = {}  # Recording
        st.session_state.wav_diagnose_df = pd.DataFrame()
        st.session_state.wav_demog_df = pd.DataFrame()  # Demographic
        st.session_state.wav_az_dict = {}   # Respiratory Cycles
        st.session_state.wav_codings = []
        st.session_state.summary_audio_information = pd.DataFrame()

    if 'wav_dataframes_audio_scaled' not in st.session_state:
        st.session_state.wav_dataframes_audio_scaled = []
        st.session_state.visualization_sr = 8000    # Downsample sample rate for visualization task
        st.session_state.wav_channels_answer = ''

    if 'resp_cycles' not in st.session_state:
        st.session_state.resp_cycles = []
        st.session_state.cycles_labels_list = []
        st.session_state.cycles_file_names = []
        st.session_state.cycles_nr = []
        st.session_state.resp_split_type = ''

        # Imputation
        st.session_state.wav_imputed_dataframes = []
        st.session_state.wav_imputation_method = ''

    if 'cluster_dataset' not in st.session_state:
        st.session_state.cluster_dataset = np.array([])

        # Subset selection
        st.session_state.subset_cycles = []
        st.session_state.subset_labels_list = []
        st.session_state.subset_cycles_nr = []
        st.session_state.subset_cycles_file_names = []
        st.session_state.cluster_subset = np.array([])

    # ############################## PAGE SELECTION ###################################
    pages = {
        "Introduction": intro,
        "Upload Files": wav_file_upload,
        "Data Entry and Extraction": data_extraction,
        "Preprocessing": preprocess,
        "Visualization Waveform": visualization_time_domain,
        "Visualization Spectrogram": visualization_timefreq_domain,
        "Extract Resp. Cycles": extract_rc,
        "Imputation": imputation,
        "Clustering": clustering,
    }

    # ######################### GENERAL HEADER ########################################
    if 'corporate_path' not in st.session_state:
        st.session_state.corporate_path = os.getcwd() + '/images/corporate/'
    logo = Image.open(st.session_state.corporate_path + 'fraunhofer_logo.png')
    col1, col2, col3, col4 = st.columns(4)
    with col4:
        st.image(logo, width=300)

    # ######################### PAGE SELECTION ########################################
    st.sidebar.title('Navigation')
    if 'page' not in st.session_state:
        st.session_state.page = "Introduction"
        st.session_state.base_dir = os.path.dirname(os.path.realpath(__file__))
    st.sidebar.radio("Select your task",
                     tuple(pages.keys()),
                     key='page',
                     on_change=_select_page())

    # ######################## FUNCTIONAL BUTTONS ######################################
    st.sidebar.subheader('File Management')
    restart_button = st.sidebar.button('Reset App')
    if restart_button:
        __delete_all_session_states()

    delete_wav_files = st.sidebar.button('Delete Progress')
    if delete_wav_files:
        __delete_progress()

    # ######################## CURRENT UPLOAD STATE ######################################
    st.sidebar.subheader("File State")
    # ### Display files which are uploaded, enable to search by name
    exp1 = st.sidebar.checkbox("**Uploaded Files**")
    if exp1:
        if len(st.session_state.wav_file_names) == 0:
            st.sidebar.warning(f"*None*")
        else:
            st.sidebar.info(f"Nr. of Files: {len(st.session_state.wav_file_names)}")
            search_term = st.sidebar.text_input('Search file:', key='st',
                                                placeholder=st.session_state.wav_file_names[0])
            wav_file_names_df = pd.DataFrame({"File Name": st.session_state.wav_file_names})
            if search_term:

                filtered_df = wav_file_names_df[
                    wav_file_names_df['File Name'].str.contains(search_term, case=False)]
                if not filtered_df.empty:
                    st.sidebar.table(filtered_df)
                else:
                    st.sidebar.markdown('*No file with given name found.*')
            else:
                st.sidebar.write(wav_file_names_df)

    # ### Display information about extracted information
    exp2 = st.sidebar.checkbox("**Data Entry and Extraction**")
    if exp2:
        if len(st.session_state.wav_rec_dict) > 0:
            st.sidebar.success("**Recording Information** uploaded.")
        else:
            st.sidebar.warning("**Recording Information** not uploaded.")
        if len(st.session_state.wav_demog_df) > 0:
            st.sidebar.success("**Demographic Information** uploaded.")
        else:
            st.sidebar.warning("**Demographic Information** not uploaded.")
        if len(st.session_state.wav_diagnose_df) > 0:
            st.sidebar.success("**Diagnose** uploaded.")
        else:
            st.sidebar.warning("**Diagnose** not uploaded.")
        if len(st.session_state.wav_az_dict) > 0:
            st.sidebar.success("**Respiratory Cycles** uploaded.")
        else:
            st.sidebar.warning("**Respiratory Cycles** not uploaded.")

    # ### Display information about preprocessing
    exp3 = st.sidebar.checkbox("**Preprocessing**")
    if exp3:
        if st.session_state.wav_channels_answer != '':
            st.sidebar.success(f"**Channel Analysis** Done. Files are: **{st.session_state.wav_channels_answer}**")
        else:
            st.sidebar.warning("**Channel Analysis** Not Performed.")
        if len(st.session_state.wav_dataframes_audio_scaled) > 0:
            st.sidebar.success("**Audio Scaling** Done.")
        else:
            st.sidebar.warning("**Audio Scaling** Not Performed.")
        if len(st.session_state.wav_imputed_dataframes) > 0:
            st.sidebar.success(f'**Imputation** with {st.session_state.wav_imputation_method}.')
        else:
            st.sidebar.warning("**Imputation** Not Performed.")

    # ### Display only when audio is split to cycles
    if len(st.session_state.resp_cycles) != 0:
        st.sidebar.info(f' {len(st.session_state.resp_cycles)} Respiratory Cycles '
                        f'extracted with **{st.session_state.resp_split_type}**.')


def _select_page():
    # ################################### INTRO PAGE ##################################################################
    if st.session_state.page == "Introduction":
        if 'visualization_path' not in st.session_state:
            st.session_state.visualization_path = os.getcwd() + '/images/corporate/'
        os.chdir(st.session_state.base_dir)

        intro.display_welcoming_info()

    # ################################### WAV FILE UPLOAD #############################################################
    elif st.session_state.page == "Upload Files":
        st.markdown('## Upload Audio Files')
        st.markdown('Here you can upload your audio files or choose from the available files. Audio file must be in '
                    'WAV format.')
        st.markdown('Uploading has to be done in one go. If the wrong files were uploaded, use the *Reset App* '
                    'button in the sidebar under *File Management* to remove the old files from the system. '
                    'After that you can upload the correct files.')
        st.markdown('You can check the sidebar at all times to get an overview of all *Uploaded Files.*')

        # Display instruction
        if len(st.session_state.wav_file_names) > 0:
            st.warning('Files already uploaded, if you want to re-upload, please use **"Reset App"** button first.')

        st.markdown('You can upload your files either with the file uploader, or use the test files provided. '
                    'Audio file must be in WAV format. ')

        col11, col12 = st.columns(2)

        with col11:
            st.markdown('##### File uploader')
            with st.form(key="upload_audio_files"):
                wav_uploader = st.file_uploader('Upload your .wav files here:',
                                                type=['wav'],
                                                accept_multiple_files=True,
                                                )
                wav_upload_button = st.form_submit_button(label='Confirm choice')
        if wav_upload_button:
            wav_file_upload.load_wav_st_uploader(wav_uploader)

        with col12:
            st.markdown('##### ICBHI-Dataset')
            with st.form(key="audio_files_ICBHI"):
                audio_options = utilities.available_audio_options()
                audio_multiselect = st.multiselect('Choose one or more files', audio_options)
                wav_button = st.form_submit_button(label='Confirm choice')
        if wav_button:
            if "All Files" in audio_multiselect:
                st.info("All Files option was chosen. Full dataset will be uploaded automatically. All other choices "
                        "are discarded.")
            wav_file_upload.load_wav_st_multiselect(audio_multiselect)

    # ################################## DATA ENTRY AND EXTRACTION ####################################################
    elif st.session_state.page == "Data Entry and Extraction":
        st.markdown("## Data Entry and Extraction")
        st.markdown('If the audio file is part of the ICBHI Dataset, use the first tab to get information made '
                    'available. An interactive table will be displayed after task competition. \n')
        st.markdown('If the wrong files were uploaded, use the *Reset App* button in the sidebar under *File '
                    'Management* to remove the old files from the system and upload new files.')
        # Display data state information
        if len(st.session_state.wav_dataframes) == 0:
            st.warning('Please **upload** your audio data.')

        st.markdown("### ICBHI Information")
        st.markdown('This information  is specific to each recording and therefore, specific to each patient. ')

        col_info_1, col_info_2, col_info_3, col_info_4 = st.columns(4)
        with col_info_1:
            st.info('**Recording Information**')
            st.write('*(based on recording/filename)*')
            st.write('- Recording index \n'
                     '- Chest location \n'
                     '- Acquisition mode \n'
                     '- Recording equipment \n')
        with col_info_2:
            st.info('**Demographic Information**')
            st.write('*(based on Patient-ID)*')
            st.markdown('- Patient-ID \n'
                        '- Age \n'
                        '- Gender \n'
                        '- Adult BMI \n'
                        '- Child Weight and Height')
        with col_info_3:
            st.info('**Diagnose**')
            st.write('*(based on Patient-ID)*')
            st.markdown('- Lung condition')
        with col_info_4:
            st.info('**Respiratory Cycles**')
            st.write('*(based on recording/filename)*')
            st.markdown('- Beginning of respiratory cycle(s) \n'
                        '- End of respiratory cycle(s) \n'
                        '- Presence/absence of crackles \n'
                        '- Presence/absence of wheezes')

        # Display task information
        if len(st.session_state.summary_audio_information) > 0:
            st.success('Information was already extracted.')

        # User Input
        with st.form(key='all_ICBHI_info'):
            st.markdown('##### Get all information in one click.')
            st.markdown('**All Options** Included.')
            wav_all_button = st.form_submit_button(label='Start')
        if wav_all_button:
            if len(st.session_state.wav_file_names) > 0:
                data_extraction.extract_patient_information()  # Extract all 4 categories
                progress_bar = st.progress(0, text='Compiling Data...')  # Compile all data to one DataFrame
                st.session_state.summary_audio_information = data_extraction.organize_all_patient_info()
                progress_bar.progress(100, text='Patient Data Compiled.')
                st.dataframe(st.session_state.summary_audio_information)
            else:
                st.warning('Please **upload** your audio files before extracting information.')

    # ################################### PREPROCESS (AUDIO SCALING) ##############################################
    elif st.session_state.page == "Preprocessing":
        st.markdown('## Preprocessing')
        st.markdown('Here you can perform audio scaling for future visualization needs.  \n\n '
                    'If the wrong files were uploaded, use the *Reset App* button in the sidebar under *File '
                    'Management* to remove the old files from the system and upload new files.')
        st.markdown('### Options:')
        st.markdown('- *Audio Scaling* - Set the amplitude range to [-1, 1] to ensure meaningful visualization.')

        # Display data state information
        if len(st.session_state.wav_dataframes) == 0:
            st.warning('Please **upload** your audio data.')

        t1 = st.tabs(["Audio Scaling"])

        with t1[0]:
            # ############################# AUDIO SCALING ########################################################
            st.markdown('#### Audio Scaling')
            st.markdown('Audio Scaling is a process that changes the level of each sample in a digital '
                        'audio signal by the same amount, such that the loudest sample reaches a specified level. '
                        'The signal-to-noise-ratio and signal shape will not be affected.')
            st.markdown('**Method:** Divide by max(abs(amplitude))')
            st.markdown('**Output:** New signal, within the range of [-1, 1]. '
                        'Either maximum or minimum will be 1 or -1.')
            st.info("Input: **UPLOADED** data.")
            with st.form(key="wav_normalize"):
                norm_button = st.form_submit_button(label="Perform audio scaling")
            if norm_button:
                if len(st.session_state.wav_dataframes) > 0:
                    if len(st.session_state.wav_dataframes_audio_scaled) > 0:
                        st.info("Audio Scaling already performed.")
                    else:
                        df_n = preprocess.audio_scaling()
                        st.session_state.wav_dataframes_audio_scaled = df_n
                        st.success('**Successful.**')
                else:
                    st.warning('Please **upload** your audio data before applying audio scaling.')

    # ################################### WAV WAVEFORM VISUALIZATION  #################################################
    elif st.session_state.page == "Visualization Waveform":
        st.markdown('## Visualization (Amplitude - Time)')
        st.markdown('Here you can plot your time series data in an amplitude-time plot. You can either display '
                    'a single file, or multiple files. For multiple files you have also the options to use a common '
                    'plot, or create a plot for each file.')
        st.markdown('Select the settings of your choice and start the plotting using the *Submit* button.')
        st.markdown('#### Options')
        st.markdown('- *File / Files* -  select the file(s) you want to plot  \n'
                    '- *All files visualization* - if _True_, uses all uploaded files, other selection will be '
                    'discarded  \n'
                    '- *Subplots* -  if _True_, every file displayed in its own plot, if _False_ one common plot is '
                    'utilized  \n'
                    '- *Time in Seconds* - timestamps in seconds  \n'
                    '- *Artificial Timestamps* - timestamps in steps / frames  \n'
                    '- *Patient Information* - textual information extracted in \'Data Entry\'  \n'
                    '- *Respiratory Cycles* - display color-coded respiratory cycles, start and end time in seconds '
                    '(only available for single file or multiple files in subplots):\n'
                    '  - *Green (N)*: Normal Sound \n'
                    '  - *Yellow (C)*: Crackle \n'
                    '  - *Orange (W)*: Wheeze \n'
                    '  - *Red (B)*: Both - mixed crackle and wheeze')

        # Display user information
        _ = utilities.check_data_presence()

        # Prep data
        file_names_list = st.session_state.wav_file_names
        if len(st.session_state.wav_dataframes_audio_scaled) > 0:
            df_list = st.session_state.wav_dataframes_audio_scaled  # Audio is downsampled and scaled accordingly
        else:
            df_list = utilities.resample_for_visual()  # Downsample before visualization

        column1, column2 = st.columns(2)

        with column1:
            # ############################### SINGLE FILE #####################################################
            with st.form(key='wav_visualization_sing_file_form'):
                st.markdown('##### Single File')
                wav_file_chooser = st.selectbox('File', file_names_list)
                opt = st.radio('Options:', ['Time in Seconds', 'Artificial Timestamps', 'Respiratory Cycles'])
                patient_opt = st.selectbox(label='Patient Information', options=['False', 'True'])
                wav_single_visual_button = st.form_submit_button(label='Submit')
        if wav_single_visual_button:
            if len(file_names_list) > 0:
                with st.spinner('Plotting...'):
                    index = file_names_list.index(wav_file_chooser)
                    current_df = df_list[index]
                    try:
                        if patient_opt == 'False':
                            p = visualization_time_domain.plotting_single_file(current_df,
                                                                               wav_file_chooser,
                                                                               opt,
                                                                               index
                                                                               )
                            st.bokeh_chart(p)
                        else:
                            p = visualization_time_domain.plotting_single_file_patientinfo(current_df,
                                                                                           wav_file_chooser,
                                                                                           opt,
                                                                                           index
                                                                                           )
                            st.bokeh_chart(p)
                    except KeyError:
                        st.warning("Please get information about the patient in *Data Entry and Extraction*.")
            else:
                st.warning('Please **upload** your audio data before plotting.')

        # ############################### MULTIPLE FILES ###################################
        with column2:
            with st.form(key='wav_visualization_multiple_files_form'):
                st.markdown('##### Multiple Files')
                wav_file_multi_chooser = st.multiselect('Files', file_names_list)
                st.write('*Recommended to display no more than 5 files in one plot for clarity*')
                wav_visualize_all_files = st.selectbox('All files visualization:', [False, True])
                wav_create_subplots = st.selectbox('Subplots:', [False, True])
                options = st.radio('Options:', ['Time in Seconds', 'Artificial Timestamps', 'Respiratory Cycles'])
                wav_multiple_visual_button = st.form_submit_button(label='Submit')
        if wav_multiple_visual_button:
            if len(file_names_list) > 0:
                if wav_create_subplots is False and options == 'Respiratory Cycles':
                    st.warning('Can not display respiratory cycles while plotting files in one common plot. Please '
                               'choose another option or plot files in subplots. ')
                else:
                    if options == 'Respiratory Cycles':
                        st.markdown('**Color Legend**')
                        legend_data = {'Green': 'Healthy', 'Yellow': ' Crackles', 'Orange': 'Wheezes', 'Red': 'Both'}
                        legend_df = pd.DataFrame([legend_data])
                        st.dataframe(legend_df, hide_index=True)

                    try:
                        if wav_visualize_all_files:
                            if wav_create_subplots:  # all files with subplots (separate)
                                with st.spinner('Plotting...'):
                                    visualization_time_domain.all_files_in_subplots(df_list, file_names_list, options)

                            else:  # all files in one common plot
                                with st.spinner('Plotting...'):
                                    visualization_time_domain.all_files_in_one_plot(df_list, file_names_list, options)

                        else:  # only some files are selected
                            if len(wav_file_multi_chooser) > 0:
                                if wav_create_subplots:  # all files with subplots (separate)
                                    with st.spinner('Plotting...'):
                                        visualization_time_domain.multiple_files_in_subplots(df_list,
                                                                                             file_names_list,
                                                                                             wav_file_multi_chooser,
                                                                                             options)
                                else:  # all files in one common plot
                                    with st.spinner('Plotting...'):
                                        visualization_time_domain.multiple_files_in_one_plot(df_list,
                                                                                             file_names_list,
                                                                                             wav_file_multi_chooser,
                                                                                             options)
                            else:
                                st.warning("Please choose files first.")
                    except KeyError:
                        st.warning("No information available. Please extract data in *Data Entry and Extraction*.")
            else:
                st.warning('Please **upload** your audio data before plotting.')

    # ################################### WAV SPECTROGRAM VISUALIZATION ###############################################
    elif st.session_state.page == "Visualization Spectrogram":
        st.markdown('## Visualization (Mel Spectrogram)')
        st.markdown('Here you can plot the mel-scaled spectrogram of your time series data. It is known that humans '
                    'do not perceive frequencies on a linear scale. We are better at detecting differences in '
                    'lower frequencies than higher frequencies. Therefore, **Mel Spectrogram** is being used, which '
                    'adjusts to this perception. **Frequency in y-axis** is therefore **logarithmic**.')
        st.markdown('Select the settings of your choice and start the plotting using the *Submit* button.')
        st.markdown('#### Options')
        st.markdown('- *File / Files* -  select the file(s) you want to plot  \n'
                    '- *All files visualization* - if _True_, uses all uploaded files, other selection will be '
                    'discarded  \n'
                    '- *Frequency Range* - desired frequency range to display, within 0-4000 Hz  \n'
                    '- *Recording Information* - textual information extracted in \'Data Entry\'  \n'
                    '- *Patient Information* - textual information extracted in \'Data Entry\'  \n'
                    '- *Respiratory Cycles* - display color-coded respiratory cycles, start and end time in seconds: \n'
                    '  - *Green (N)*: Normal Sound \n'
                    '  - *Yellow (C)*: Crackle \n'
                    '  - *Orange (W)*: Wheeze \n'
                    '  - *Red (B)*: Both - mixed crackle and wheeze')

        # Just display the visual output
        _ = utilities.check_data_presence()

        # Prep time series data
        file_names_list = st.session_state.wav_file_names
        if len(st.session_state.wav_dataframes_audio_scaled) > 0:  # Audio is resampled and scaled accordingly
            df_list = st.session_state.wav_dataframes_audio_scaled
        else:  # Resample before visualization
            df_list = utilities.resample_for_visual()

        column1, column2 = st.columns(2)
        # ############################### SINGLE FILE ###################################
        with column1:
            with st.form(key='wav_specgram_sing_file_form'):
                st.markdown('##### Single File')
                wav_file_chooser = st.selectbox('File', file_names_list)
                f_min_input = np.int32(st.number_input(label='Minimum Frequency in Hz', min_value=0, max_value=3999,
                                                       value=60))
                f_max_input = np.int32(st.number_input(label='Maximum Frequency in Hz', min_value=1, max_value=4000,
                                                       value=2500))
                checkbox_rec_info = st.checkbox('Recording Information', key='rec_s')
                checkbox_pat_info = st.checkbox('Patient Information', key='demography_s')
                checkbox_az = st.checkbox('Respiratory Cycles', key='az_s')
                wav_single_specgram_button = st.form_submit_button(label='Submit')

        if wav_single_specgram_button:
            if len(file_names_list) > 0:
                # Frequency range check
                if f_min_input > f_max_input:
                    st.warning("Minimum frequency should be smaller than maximum frequency displayed in the y-axis. "
                               "Please choose other values.")
                else:
                    index = file_names_list.index(wav_file_chooser)
                    current_df = df_list[index]
                    try:
                        with st.spinner('Plotting Mel-Spectrogram...'):
                            visualization_timefreq_domain.wav_specgram_single_file_detail(current_df, wav_file_chooser,
                                                                                          index, checkbox_rec_info,
                                                                                          checkbox_pat_info,
                                                                                          checkbox_az,
                                                                                          f_min_input, f_max_input)
                    except KeyError:
                        st.warning("Please get information about the patient in *Data Entry and Extraction*.")
            else:
                st.warning('Please **upload** your audio data before plotting.')
        # ############################### MULTIPLE FILES ###################################
        with column2:
            with st.form(key='wav_specgram_multiple_files_form'):
                st.markdown('##### Multiple Files')
                wav_file_multi_chooser = st.multiselect('Files', file_names_list)
                wav_spec_all_files = st.selectbox('All files visualization:', [False, True])
                wav_az_state = st.selectbox('Respiratory Cycles:', [False, True])
                wav_multiple_specgram_button = st.form_submit_button(label='Submit')
        if wav_multiple_specgram_button:
            if len(file_names_list) > 0:
                try:
                    if wav_spec_all_files is True:
                        with st.spinner('Plotting Mel-Spectrogram...'):
                            visualization_timefreq_domain.wav_specgram_all_files(file_names_list, df_list, wav_az_state)
                    else:
                        if len(wav_file_multi_chooser) > 0:
                            with st.spinner('Plotting Mel-Spectrogram...'):
                                visualization_timefreq_domain.wav_specgram_multiple_files(wav_file_multi_chooser,
                                                                                          file_names_list,
                                                                                          df_list, wav_az_state)
                        else:
                            st.warning("Please choose from files first.")
                except KeyError:
                    st.warning("Please get information about the patient in *Data Entry and Extraction*.")
            else:
                st.warning('Please **upload** your audio data before plotting.')

    # ################################### SPLIT TO CYCLES #############################################################
    elif st.session_state.page == "Extract Resp. Cycles":
        st.markdown('## Extract Respiratory Cycles')
        st.markdown('Here you can extract respiratory cycles from each recording, in order to apply clustering '
                    'with the k-means algorithms. To do this, you first need to get the relevant information in '
                    '*Data Entry and Extraction.*')

        st.markdown('Based on start/end time *original cycles* are extracted. For some tasks, equal-sized cycles are '
                    'required. For this purpose, one can apply the option *\'Fix to desired duration\'*.')

        st.markdown('#### Options')
        st.markdown('- *Fix to desired duration* - resample so that the desired length given in seconds is '
                    'achieved  \n'
                    '- *Original:* - extract raw respiratory cycles  \n '
                    '*Note:* The respiratory cycle durations in the ICBHI dataset range from 0.2 to 16.1 seconds.')

        # Hint for the user to upload data
        if len(st.session_state.wav_dataframes) == 0:
            st.warning('Please **upload** your audio data.')
        else:
            st.info('Input: **UPLOADED** data')

        # Hint for the user if the respiratory cycle extraction task was already performed
        if len(st.session_state.resp_cycles) > 0:
            st.warning('Extraction was already performed. **Rerun** with new option if needed.')

        with st.form(key='extract_rc'):
            st.markdown('##### Extract Resp. Cycles')
            desired_time_seconds = st.number_input(label='Time in seconds (fixed, optional)',
                                                   min_value=0.0, value=4.0, step=0.1)
            split_type = st.radio(label='Options', options=['Fix to desired duration', 'Original'])
            extract_rc_button = st.form_submit_button('Submit.')

        if extract_rc_button:
            if len(st.session_state.wav_dataframes) > 0:
                try:
                    st.session_state.resp_split_type = split_type
                    if split_type == 'Fix to desired duration':
                        extract_rc.extract_resp_cycles_fixed_length(st.session_state.wav_dataframes,
                                                                    desired_time_seconds)
                        st.success(f'Extraction successfully completed.')
                        st.success(f'Nr of cycles: **{len(st.session_state.resp_cycles)}**')

                        # Perform length check
                        extract_rc.check_cycles_length(desired_time_seconds)

                    elif split_type == 'Original':
                        extract_rc.extract_resp_cycles(st.session_state.wav_dataframes)
                        st.success(f'Extraction successfully completed.')
                        st.success(f'Nr of cycles: **{len(st.session_state.resp_cycles)}**')

                except KeyError:
                    st.warning('Get information about respiratory cycles in **Data Entry and Extraction** before '
                               'proceeding.')
            else:
                st.warning('Please **upload** your audio data before proceeding.')

    #  #################################### IMPUTATION ################################################################
    elif st.session_state.page == "Imputation":
        st.markdown('## Imputation')
        st.markdown(
            'Missing values are very common in medical time series. In order that various analysis methods '
            'can nevertheless provide good results (or work at all), these must be replaced by other values. '
            'Since only numerical values can be imputed, it will be checked if the data includes columns of '
            'other types like *objects* and deletes these columns automatically.')
        st.markdown(
            'We offer two imputations methods. *Local*, where only values within a file are used for the '
            'calculation, and *global*, where values from all files are used for the calculation of '
            'missing values. If you are unsure if every dimension of every file has at least one value '
            '*global* imputations is recommended.')
        st.markdown('Select the settings of your choice and start the imputations using the *Submit* button.')
        st.markdown('#### Options')
        st.markdown('*Method* - select the imputations method for calculating missing values')

        # Hint for the user about data input
        if len(st.session_state.resp_cycles) == 0:
            st.warning("Please **extract respiratory cycles**.")
        else:
            st.info(" Input: **RESPIRATORY CYCLES** data.")

        # Hint for the user if the task was already performed
        if len(st.session_state.wav_imputation_method) > 0:
            st.warning(
                f'Data already imputed with {st.session_state.wav_imputation_method}. If you want to change, you '
                f'can rerun with the settings of your choice.')

        col_imp_1, col_imp_2 = st.columns(2)

        with col_imp_1:  # Local
            with st.form(key='wav_sktime_imputation'):
                st.markdown('##### Local')
                wav_imputation_method = st.selectbox('Method:',
                                                     ['linear',
                                                      # 'drift',
                                                      'nearest',
                                                      # #'constant',
                                                      'mean',
                                                      'median',
                                                      'backfill',
                                                      'pad',
                                                      # 'random'
                                                      ])
                wav_local_imputation_submit_button = st.form_submit_button('Submit')
        if wav_local_imputation_submit_button:
            if len(st.session_state.resp_cycles) > 0:
                st.session_state.wav_imputed_dataframes = imputation.local_imputation(st.session_state.resp_cycles,
                                                                                      wav_imputation_method)
                st.session_state.wav_imputation_method = 'local (' + wav_imputation_method + ')'
                st.success(f'##### Imputation of {len(st.session_state.wav_imputed_dataframes)} cycles '
                           f'with method {st.session_state.wav_imputation_method} successful.')
            else:
                st.warning('Please **extract respiratory cycles** before imputation.')

        with col_imp_2:  # Global
            with st.form(key='wav_own_imputation'):
                st.markdown('##### Global')
                wav_imputation_method = st.selectbox('Method:',
                                                     ['mean',
                                                      'median',
                                                      'zero'])
                wav_global_imputation_submit_button = st.form_submit_button('Submit')
        if wav_global_imputation_submit_button:
            if len(st.session_state.resp_cycles) > 0:
                st.session_state.wav_imputed_dataframes = imputation.global_imputation(st.session_state.resp_cycles,
                                                                                       wav_imputation_method)
                st.session_state.wav_imputation_method = 'global (' + wav_imputation_method + ')'
                st.success(f'##### Imputation of {len(st.session_state.wav_imputed_dataframes)} cycles '
                           f'with method {st.session_state.wav_imputation_method} successful.')
            else:
                st.warning('Please **extract respiratory cycles** before imputation.')

    # ################################### WAV CLUSTERING ##############################################################
    elif st.session_state.page == "Clustering":
        st.markdown('## Clustering')
        st.markdown('Clustering is the task of grouping a set of objects in a way that objects of the same group '
                    'are more similar to each other than to objects of any other group.')
        st.markdown('Select the settings of your choice and start the clustering using the *Submit* button.')
        st.markdown('#### Options')
        st.markdown(' - *Data* - choose the data that should be used for the clustering')
        st.markdown('##### Prep for clustering')
        st.markdown('- *Normalization* - do not use at all, use Min-Max to fit within [-1,1] range or apply '
                    'Z-Score normalization to remove the mean and divide by standard deviation  \n'
                    '- *PAA* - do not aggregate or use PAA with desired number of segments as input')
        st.markdown('##### TimeSeriesKMeans')
        st.markdown(' - *Distance Function* - the function used to calculate the distances between instances  \n'
                    ' - *k* - the number of groups that the data should be divided into  \n'
                    ' - *Init* - method used to initialize the model  \n'
                    ' - *Apply Sakoe-Chiba Band* - set *True* if desired to be applied (only for DTW)  \n'
                    ' - *Sakoe-Chiba Band* - radius to be applied, relative to time series length in %')
        st.markdown('##### KernelMeans')
        st.markdown('*Kernel Function* - Kernel function used to map data')

        # Hint for the data
        if len(st.session_state.resp_cycles) == 0:
            st.warning("Please **extract respiratory cycles**.")

        # Define data selection
        if len(st.session_state.wav_imputed_dataframes) > 0:
            wav_clustering_data = {
                'Respiratory Cycles': st.session_state.resp_cycles,
                'Imputed Cycles': st.session_state.wav_imputed_dataframes,

            }

        else:
            wav_clustering_data = {
                'Respiratory Cycles': st.session_state.resp_cycles
            }

        # Two tabs: Uploaded files / Subset
        st.markdown('If the full dataset is uploaded, a subset can be chosen to use clustering.')

        tab1, tab2 = st.tabs(['Continue with uploaded files.', 'Choose subset.'])

        with tab1:
            data_selection = st.selectbox('Data', wav_clustering_data, key='no_subset')
            # ######## Prep Dataset

            with st.form(key='Normalize_PAA'):
                st.markdown('##### Prep Dataset - Normalization & Dimensionality Reduction')
                normalization_type = st.radio(label='Normalization', options=['None', 'Min-Max', 'Z-Score'])
                agg_type = st.radio(label='Type of dimensionality reduction', options=['None', 'PAA'])
                n_seg = st.number_input(label='Nr. of Segments', min_value=1)
                paa_button = st.form_submit_button('Submit')
            if paa_button:
                if len(st.session_state.resp_cycles) > 0:
                    st.session_state.cluster_dataset = clustering.prep_clust_data(wav_clustering_data[data_selection],
                                                                                  agg_type, np.int32(n_seg),
                                                                                  normalization_type)
                    st.success('Preprocessing complete!')
                else:
                    st.warning("Please **extract respiratory cycles** before starting the dataset preparation.")

            #  ######### Clustering
            st.markdown('---')
            col1, col2, col3 = st.columns(3)
            with col1:
                with st.form(key='w_TimeSeriesKMeans'):
                    st.markdown('##### TimeSeriesKMeans')
                    distance_function = st.selectbox('Distance Function', ['Euclidean', 'DTW'])
                    num_clusters = st.selectbox('*k* (number of clusters)',
                                                list(range(2, len(st.session_state.resp_cycles))))
                    initial_centroid_input = st.selectbox('Init:', ['k-means++'])
                    sakoe_state = st.selectbox('Apply Sakoe-Chiba Band', [False, True])
                    sakoe = st.number_input(label='Sakoe-Chiba Band (% of time series length)',
                                            min_value=0, max_value=50, step=1, value=20)
                    kmeans_submit_button = st.form_submit_button('Submit')
            if kmeans_submit_button:
                if st.session_state.cluster_dataset.size != 0:
                    clustering.kmeans_clustering(st.session_state.cluster_dataset,
                                                 distance_function,
                                                 sakoe_state,
                                                 sakoe,
                                                 num_clusters,
                                                 initial_centroid_input,
                                                 st.session_state.cycles_labels_list,
                                                 st.session_state.cycles_file_names,
                                                 st.session_state.cycles_nr)
                else:
                    st.warning('Please **prepare the dataset** first using the form "Prep Dataset - Normalization & '
                               'Dimensionality Reduction".')

            with col2:
                with st.form(key='w_KernelMeans'):
                    st.markdown('##### KernelMeans')
                    kernel_function = st.selectbox("Kernel Function",
                                                   ['gak',
                                                    # 'additive_chi2', # negative values
                                                    # 'chi2', # negative values
                                                    'linear',
                                                    'poly',
                                                    'polynomial',
                                                    'rbf',
                                                    'laplacian',
                                                    'sigmoid',
                                                    'cosine'])
                    num_clusters = st.selectbox('Cluster', list(range(2, 10)))
                    wav_kernel_submit_button = st.form_submit_button('Submit')
            if wav_kernel_submit_button:
                if st.session_state.cluster_dataset.size != 0:
                    clustering.kernel_means_clustering(st.session_state.cluster_dataset,
                                                       kernel_function,
                                                       num_clusters,
                                                       st.session_state.cycles_labels_list,
                                                       st.session_state.cycles_file_names,
                                                       st.session_state.cycles_nr)
                else:
                    st.warning('Please **prepare the dataset** first using the form "Prep Dataset - Normalization & '
                               'Dimensionality Reduction".')

            with col3:
                with st.form(key='w_KShape'):
                    st.markdown('#### KShape')
                    num_clusters = st.selectbox('Cluster', list(range(2, 10)))
                    wav_kshape_submit_button = st.form_submit_button('Submit')
            if wav_kshape_submit_button:
                if st.session_state.cluster_dataset.size != 0:
                    clustering.kshape_clustering(st.session_state.cluster_dataset,
                                                 num_clusters,
                                                 st.session_state.cycles_labels_list,
                                                 st.session_state.cycles_file_names,
                                                 st.session_state.cycles_nr)
                else:
                    st.warning('Please **prepare the dataset** first using the form "Prep Dataset - Normalization & '
                               'Dimensionality Reduction".')

        with tab2:  # ########### SUBSET CHOICE
            if len(wav_clustering_data[data_selection]) != 6898:
                st.warning('Please upload the full dataset and extract respiratory cycles accordingly.')

            else:
                if len(st.session_state.subset_cycles) > 0:
                    st.info(f'Subset of {len(st.session_state.subset_cycles)} cycles already extracted.')

                data_selection_subset = st.selectbox('Data', wav_clustering_data, key='subset_select')
                clust_data = wav_clustering_data[data_selection_subset]

                with st.form(key="subset_selection"):
                    st.markdown("##### Select subset")
                    subset_size = st.selectbox(label='Subset size to use in %',
                                               options=[5, 10, 15])
                    subset_nr = st.selectbox(label='Subset Nr.', options=[i + 1 for i in range(2)])
                    subset_button = st.form_submit_button('Extract Subset')
                if subset_button:
                    st.session_state.subset_cycles, st.session_state.subset_labels_list, \
                        st.session_state.subset_cycles_nr, st.session_state.subset_cycles_file_names = \
                        subset_clustering.choose_subset_size_nr(subset_size, subset_nr, clust_data)

                # ######## Prep Dataset
                st.markdown('---')
                with st.form(key='Normalize_PAA_subset'):
                    st.markdown('##### Prep Dataset - Normalization & Dimensionality Reduction')
                    normalization_type_subset = st.radio(label='Normalization',
                                                         options=['None', 'Min-Max', 'Z-Score'])
                    agg_type_subset = st.radio(label='Type of dimensionality reduction', options=['None', 'PAA'])
                    n_seg_subset = st.number_input(label='Nr. of Segments', min_value=1)
                    paa_button_subset = st.form_submit_button('Submit')
                if paa_button_subset:
                    if len(st.session_state.subset_cycles) > 0:
                        st.write('Nr. of cycles: ', len(st.session_state.subset_cycles))
                        st.session_state.cluster_subset = clustering.prep_clust_data(st.session_state.subset_cycles,
                                                                                     agg_type_subset,
                                                                                     np.int32(n_seg_subset),
                                                                                     normalization_type_subset)
                    else:
                        st.warning('Please **choose subset** before prepping data.')

                #  ######### Clustering
                st.markdown('---')
                col1, col2, col3 = st.columns(3)
                with col1:
                    with st.form(key='w_TimeSeriesKMeans_subset'):
                        st.markdown('##### TimeSeriesKMeans')
                        distance_function = st.selectbox('Distance Function', ['Euclidean', 'DTW'])
                        num_clusters = st.selectbox('*k* (number of clusters)',
                                                    list(range(2, len(st.session_state.subset_cycles))))
                        initial_centroid_input = st.selectbox('Init:', ['k-means++'])
                        sakoe_state = st.selectbox('Apply Sakoe-Chiba Band', [False, True])
                        sakoe = st.number_input(label='Sakoe-Chiba Band (% of time series length)',
                                                min_value=0, max_value=50, step=1, value=20)
                        kmeans_submit_button = st.form_submit_button('Submit')
                if kmeans_submit_button:
                    if st.session_state.cluster_subset.size != 0:
                        clustering.kmeans_clustering(st.session_state.cluster_subset,
                                                     distance_function,
                                                     sakoe_state,
                                                     sakoe,
                                                     num_clusters,
                                                     initial_centroid_input,
                                                     st.session_state.subset_labels_list,
                                                     st.session_state.subset_cycles_file_names,
                                                     st.session_state.subset_cycles_nr)
                    else:
                        st.warning(
                            'Please **prepare the dataset** first using the form "Prep Dataset - Normalization & '
                            'Dimensionality Reduction".')

                with col2:
                    with st.form(key='w_KernelMeans_subset'):
                        st.markdown('##### KernelMeans')
                        kernel_function = st.selectbox("Kernel Function",
                                                       ['gak',
                                                        # 'additive_chi2', # negative values
                                                        # 'chi2', # negative values
                                                        'linear',
                                                        'poly',
                                                        'polynomial',
                                                        'rbf',
                                                        'laplacian',
                                                        'sigmoid',
                                                        'cosine'])
                        num_clusters = st.selectbox('Cluster', list(range(2, 10)))
                        wav_k_means_submit_button = st.form_submit_button('Submit')
                if wav_k_means_submit_button:
                    if st.session_state.cluster_subset.size != 0:
                        clustering.kernel_means_clustering(st.session_state.cluster_subset,
                                                           kernel_function,
                                                           num_clusters,
                                                           st.session_state.subset_labels_list,
                                                           st.session_state.subset_cycles_file_names,
                                                           st.session_state.subset_cycles_nr)
                    else:
                        st.warning('Please **prepare the dataset** first using the form "Prep Dataset - Normalization '
                                   '& Dimensionality Reduction".')

                with col3:
                    with st.form(key='w_KShape_subset'):
                        st.markdown('#### KShape')
                        num_clusters = st.selectbox('Cluster', list(range(2, 10)))
                        wav_kshape_submit_button = st.form_submit_button('Submit')
                if wav_kshape_submit_button:
                    if st.session_state.cluster_subset.size != 0:
                        clustering.kshape_clustering(st.session_state.cluster_subset,
                                                     num_clusters,
                                                     st.session_state.subset_labels_list,
                                                     st.session_state.subset_cycles_file_names,
                                                     st.session_state.subset_cycles_nr)
                    else:
                        st.warning('Please **prepare the dataset** first using the form "Prep Dataset - Normalization '
                                   '& Dimensionality Reduction".')


def __delete_all_session_states():
    for key in st.session_state.keys():
        del st.session_state[key]
    # rerun() so that possible output is also updated correctly
    st.rerun()


def __delete_progress():
    if 'wav_dataframes_audio_scaled' in st.session_state:
        st.session_state.wav_dataframes_audio_scaled = []
        st.session_state.wav_channels_answer = ''

    if 'resp_cycles' in st.session_state:
        st.session_state.resp_cycles = []
        st.session_state.cycles_labels_list = []
        st.session_state.cycles_file_names = []
        st.session_state.cycles_nr = []
        st.session_state.resp_split_type = ''

        # Imputation
        st.session_state.wav_imputed_dataframes = []
        st.session_state.wav_imputation_method = ''

    if 'cluster_dataset' in st.session_state:
        st.session_state.cluster_dataset = np.array([])

        # Subset selection
        st.session_state.subset_cycles = []
        st.session_state.subset_labels_list = []
        st.session_state.subset_cycles_nr = []
        st.session_state.subset_cycles_file_names = []
        st.session_state.cluster_subset = np.array([])

    del st.session_state.page
    # rerun() so that possible output is also updated correctly
    st.rerun()


if __name__ == '__main__':
    main()
