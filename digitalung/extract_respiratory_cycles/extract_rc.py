"""Collection of Functions used to split audio files of lung sounds to separate respiratory cycles.


Important Note: Output is a list of respiratory cycle. Each cycle is a dataframe with two columns:
time_steps - has time stamps in seconds, time_steps - has amplitude values

They are prepped in such a format to make the other integration of TSLibMed components easier and smoother.
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import streamlit as st


def extract_resp_cycles(df_list):
    """
    Extract respiratory cycles of original length based on start/end.
    Saves in session state:
     - resp_cycles - list containing cycles in dataframe format
     - cycles_labels_list - list of label per cycle
     - cycles_file_names - list of audio file name corresponding to cycle
     - cycles_nr - list of cycle position within the audio
    :param df_list: List of audio files in dataframe format
    """

    progress_bar = st.progress(text='Extracting ... ', value=0)

    cycles_df_list = []
    cycles_labels_list = []
    cycles_file_names = []
    cycles_nr = []

    # extract cycle based on start and end
    for i, audio in enumerate(df_list):
        name = st.session_state.wav_file_names[i]
        resp_info = st.session_state.wav_az_dict[i]
        for j, r in resp_info.iterrows():
            cycle = {}
            start = int(r['start'] * st.session_state.librosa_sr)
            end = int(r['end'] * st.session_state.librosa_sr)
            cycle_col = audio.iloc[start:end, 1]
            cycle_col = np.array(cycle_col)

            t = audio.iloc[start:end, 0]
            cycle_raw = pd.DataFrame({'time_steps': t, 'dim_0': cycle_col})

            # Get Label of Presence C/W
            coding_map = {
                (0, 0): 'N',
                (1, 0): 'C',
                (0, 1): 'W',
                (1, 1): 'B'
            }
            crackles = r['crackles']
            wheezes = r['wheezes']
            presence = coding_map[(crackles, wheezes)]

            # Append all information to dict
            cycle['data'] = cycle_raw
            cycle['start'] = start
            cycle['end'] = end

            cycles_df_list.append(cycle_raw)
            cycles_labels_list.append(presence)
            cycles_file_names.append(name)
            cycles_nr.append(j + 1)
        progress_bar.progress(value=(i / (len(df_list))), text='Extracting ...')

    st.session_state.resp_cycles = cycles_df_list
    st.session_state.cycles_labels_list = cycles_labels_list
    st.session_state.cycles_file_names = cycles_file_names
    st.session_state.cycles_nr = cycles_nr

    progress_bar.progress(value=100, text='Done.')


def extract_resp_cycles_fixed_length(df_list, desired_time_seconds):
    """
    Extract respiratory cycles and achieve fixed-length by resampling to a desired size.
    Saves in session state:
     - resp_cycles - list containing cycles in dataframe format
     - cycles_labels_list - list of label per cycle
     - cycles_file_names - list of audio file name corresponding to cycle
     - cycles_nr - list of cycle position within the audio

    :param df_list: List of audio files in dataframe format
    :param desired_time_seconds: target time in seconds
    """

    progress_bar = st.progress(text='Extracting ... ', value=0)

    cycles_df_list = []
    cycles_labels_list = []
    cycles_file_names = []
    cycles_nr = []

    # extract cycle based on start and end
    for i, audio in enumerate(df_list):
        name = st.session_state.wav_file_names[i]
        resp_info = st.session_state.wav_az_dict[i]
        for j, r in resp_info.iterrows():
            cycle = {}
            start = int(r['start'] * st.session_state.librosa_sr)
            end = int(r['end'] * st.session_state.librosa_sr)
            cycle_col = np.array(audio.iloc[start:end, 1])

            if len(cycle_col) != desired_time_seconds * st.session_state.librosa_sr:        # fix to desired length
                cycle_data = signal.resample(cycle_col, np.int64(desired_time_seconds * st.session_state.librosa_sr),
                                             window='hann')
                times_col = np.linspace(0, len(cycle_data) - 1, num=len(cycle_data))
                cycle_df = pd.DataFrame({'time_steps': times_col, 'dim_0': cycle_data})

            else:
                times_col = np.linspace(0, len(cycle_col) - 1, num=len(cycle_col))
                cycle_df = pd.DataFrame({'time_steps': times_col, 'dim_0': cycle_col})

            # Get Label of Presence C/W
            coding_map = {
                (0, 0): 'N',
                (1, 0): 'C',
                (0, 1): 'W',
                (1, 1): 'B'
            }
            crackles = r['crackles']
            wheezes = r['wheezes']
            presence = coding_map[(crackles, wheezes)]

            # Append all information to dict
            cycle['data'] = cycle_df
            cycle['start'] = start
            cycle['end'] = end
            cycles_df_list.append(cycle_df)
            cycles_labels_list.append(presence)
            cycles_file_names.append(name)
            cycles_nr.append(j + 1)

        progress_bar.progress(value=i / (len(df_list)), text='Extracting ...')

    st.session_state.resp_cycles = cycles_df_list  # list of dataframes time-amplitude
    st.session_state.cycles_labels_list = cycles_labels_list  # label C;W;N;B
    st.session_state.cycles_file_names = cycles_file_names  # corresponding file name
    st.session_state.cycles_nr = cycles_nr  # cycle position ( 0,1,...n) for n cycles in audio file

    progress_bar.progress(value=100, text='Done.')


def check_cycles_length(goal_value_seconds):
    """
    Checks if all respiratory cycles have a specific duration and outputs for the user error/success messages.
    :param goal_value_seconds: Desired duration to check the respiratory cycles, unit in seconds.
    """
    a = 0
    for cycle in st.session_state.resp_cycles:
        if len(cycle) != (np.int64(goal_value_seconds * st.session_state.librosa_sr)):
            st.error(f'Wrong Length of {len(cycle) / st.session_state.librosa_sr} '
                     f'instead of {goal_value_seconds} detected.')
            a = 1
    if a == 0:
        st.success('No wrong length detected.')
