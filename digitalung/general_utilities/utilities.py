"""Collection of Utility Functions for WAV-files

This module contains a collection of functions that are used in one or more than one other module.
"""

import math
import os

import librosa
import numpy as np
import pandas as pd
import streamlit as st
from tslearn.utils import from_sktime_dataset, to_sktime_dataset


@st.cache_data
def available_audio_options():
    """ It creates a sorted list of all available audio files in the repository, that can be uploaded.

    :return: A list of sorted file names
    :rtype: list

    """
    test_audio_options = []
    for file in os.listdir('./ICBHI_dataset/audio_and_txt'):
        if file.endswith(".wav"):
            test_audio_options.append(file)
    os.chdir(st.session_state.base_dir)
    test_audio_options = sorted(test_audio_options)
    test_audio_options.insert(0, "All Files")
    return test_audio_options


def check_data_presence():
    """This function check what is saved in session state and decides which time series data is going to be used.
    It returns the time series list and displays an informative message for the user.

        :return: A list of dataframes
        :rtype: list
        """
    df_list = []
    if len(st.session_state.wav_dataframes) == 0:
        st.warning('Please **upload** your audio data.')
    elif len(st.session_state.wav_dataframes) > 0 and len(st.session_state.wav_dataframes_audio_scaled) == 0:
        st.info("Not normalized. Input: **UPLOADED** data.")
        df_list = st.session_state.wav_dataframes
    elif len(st.session_state.wav_dataframes_audio_scaled) > 0:
        st.info("Input: **SCALED** data.")
        df_list = st.session_state.wav_dataframes_audio_scaled
    return df_list


def __get_demog_info(pat_id):
    """This function returns demographic information corresponding to the patient-ID.
    The information is obtained from the session states.

    :param pat_id: Patient-ID
    :type pat_id: int

    :return: A tuple containing information about the patient - age, sex, adult bmi, child weight and child height.
    :rtype: Tuple[float, str, float, float, float]
    """
    demog_df = st.session_state.wav_demog_df
    age = demog_df.loc[demog_df['patient_id'] == int(pat_id), 'age'].iloc[0]
    sex = demog_df.loc[demog_df['patient_id'] == int(pat_id), 'sex'].iloc[0]
    adult_bmi = demog_df.loc[demog_df['patient_id'] == int(pat_id), 'adult_bmi'].iloc[0]
    ch_weight = demog_df.loc[demog_df['patient_id'] == int(pat_id), 'ch_weight'].iloc[0]
    ch_height = demog_df.loc[demog_df['patient_id'] == int(pat_id), 'ch_height'].iloc[0]
    return age, sex, adult_bmi, ch_weight, ch_height


def get_rec_info_patient(index):
    """This function returns recording information based on recording name.
    The information is obtained from the coded name.

    :param index: Identifier for file
    :type index: int

    :return: A DataFrame containing information about the recording - recording_index, chest_location, acquisition_mode,
    recording_equipment.
    :rtype: DataFrame
    """
    name = st.session_state.wav_file_names[index]
    rec_index = st.session_state.wav_rec_dict[index]['recording_index']
    rec_loc = st.session_state.wav_rec_dict[index]['chest_location']
    acq_m = st.session_state.wav_rec_dict[index]['acquisition_mode']
    mic = st.session_state.wav_rec_dict[index]['recording_equipment']
    rec_data = {
        'Category': ['File', 'Recording Index', 'Recording Location', 'Channels', 'Equipment'],
        'Information': [name, rec_index, rec_loc, acq_m, mic]
    }
    rec_df = pd.DataFrame(rec_data)
    return rec_df


def get_pat_info_patient(index):
    """This function returns patient information based on patient-ID.
        The information is obtained from session state.

        :param index: Identifier for file
        :type index: int

        :return: A DataFrame containing information about the patient - 'Patient-ID', 'Diagnose', 'Age (years)',
        'Gender', 'Adult BMI (kg/m2)', 'Child-Weight (kg)', 'Child-Height (cm)'
        :rtype: pd.DataFrame
        """
    pat_id = st.session_state.wav_rec_dict[index]['patient_number']
    diagnose_df = st.session_state.wav_diagnose_df
    diagnose = diagnose_df.loc[diagnose_df['patient_id'] == int(pat_id), 'diagnosis'].iloc[0]
    age, sex, adult_bmi, ch_weight, ch_height = __get_demog_info(pat_id)
    pat_data = {
        'Category': ['Patient-ID', 'Diagnose', 'Age (years)', 'Gender', 'Adult BMI (kg/m2)', 'Child-Weight (kg)',
                     'Child-Height (cm)'],
        'Patient-Information': [pat_id, diagnose, age, sex, adult_bmi, ch_weight, ch_height]
    }
    pat_df = pd.DataFrame(pat_data)
    pat_df = pat_df.dropna(subset=['Patient-Information'])
    return pat_df


def resample_for_visual():
    """
    It downsamples the audio file to 8kHz sample rate before visualization to reduce data points.
    It is used if the audio scaling is not performed.
    @return:
    """
    df_resampled = []
    for i, df in enumerate(st.session_state.wav_dataframes):
        ampl = df.iloc[:, 1].values
        ampl_col = librosa.resample(ampl, orig_sr=st.session_state.librosa_sr,
                                    target_sr=st.session_state.visualization_sr, res_type='soxr_hq')
        time_col = np.linspace(0, len(ampl_col) / st.session_state.visualization_sr, num=len(ampl_col))
        downsampled_df = pd.DataFrame({'time_steps': time_col, 'dim_0': ampl_col})
        df_resampled.append(downsampled_df)
    return df_resampled


def round_up_f_dec(x):
    dec_part = math.ceil((x - int(x)) * 10) / 10
    return int(x) + dec_part


def from_wav_df_list_to_sktime_df(df_list):
    frames = [pd.DataFrame({"dim_" + str(i): [pd.Series(df[col])] for (i, col) in enumerate(df.columns)}) for df in
              df_list]
    final = pd.concat(frames, ignore_index=True)
    st.write("Sktime final from frames: df shape: ", final.shape)
    return final


def from_sktime_df_to_wav_df_list(sktime_df):  # #* length: files number
    df_list = []
    for i in range(len(sktime_df)):
        row_i = sktime_df.iloc[i]
        d = {}

        for col in sktime_df:
            col_values = row_i[col].values
            d.update({col: col_values})

        df = pd.DataFrame.from_dict(d)  # #* type(df): pandas.core.frame.DataFrame
        df_list.append(df)
    return df_list


def from_sktime_df_to_tslearn_ndarray(sktime_df):
    return from_sktime_dataset(sktime_df)


def from_csv_df_list_to_sktime_df(df_list):
    # pandas df that is compatible with the sktime library
    frames = [pd.DataFrame({"dim_" + str(i): [pd.Series(df[col])] for (i, col) in enumerate(df.columns)}) for df in
              df_list]
    final = pd.concat(frames, ignore_index=True)
    return final


def from_tslearn_ndarray_to_sktime_df(tslearn_ndarray):
    """return from_3d_numpy_to_nested(from_tslearn_ndarray_to_sktime_ndarray(tslearn_ndarray))"""
    return to_sktime_dataset(tslearn_ndarray)


def missing_value_count_in_nested_pd_series(nested_pd_series, verbose=False):
    count = 0

    for i in range(len(nested_pd_series)):
        count += nested_pd_series[i].isnull().sum()

    if verbose:
        st.write("Missing value count:", count)

    return count


def str_percentage_of_elements_in_clusters(num_instances, L):
    L = np.array(L).ravel()
    ratios = [round(num_instances_in_a_cluster / (num_instances * 1.0), 4) for num_instances_in_a_cluster in L]
    res = ""

    for i, ratio in enumerate(ratios):
        if i == len(ratios) - 1:
            res = res + str(round(ratio * 100, 2)) + "%"
        else:
            res = res + str(round(ratio * 100, 2)) + "%, "

    return res
