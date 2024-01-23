"""Collection of Functions used in the page for uploading files.
"""

import os
import wave
from copy import deepcopy

import librosa
import numpy as np
import pandas as pd
import streamlit as st


def _open_wav(path):
    """This function returns the audio array and parameters of a WAV file, when the path to the file is passed.

    :param path: Path to wav-file can be streamlit file uploader or string from test options
    :type path: BytesIO

    :return: A tuple containing the numpy array of wav-file and parameters (nchannels, sampwidth, framerate,
    nframes, comptype, compname).
    :rtype: tuple[np.array, wave.Wave_params]
    """
    p1 = deepcopy(path)
    with wave.open(p1, "rb") as w:
        params = w.getparams()
    p2 = deepcopy(path)
    audio_arr, _ = librosa.load(p2, sr=st.session_state.librosa_sr, mono=True)
    # "mono=True" to continue working only with one channel.
    # ICBHI Dataset has mono files, so that the rest of the tasks are also tailored to mono format.
    # Multichannel approach could be an approach for future versions.
    return audio_arr, params


# To be used in multichannel format, set mono=False in _open_wav(path) before using.
# def _wav_array_to_wav_dataframe(audio_arr, params):
#     """
#     Converts audio array read with librosa, mono-channel to dataframe with integrated time steps in seconds.
#     :param audio_arr: Audio array read with librosa
#     :param params: wave object containing all parameters of WAV file
#     :return: wav_df - audio DataFrame with columns: time_steps, dim_0
#     """
#     if params.nchannels == 1:
#         wav_df = pd.DataFrame(audio_arr, columns=[f"dim_0"])  # Audio data
#         wav_df.insert(0, "time_steps", np.linspace(0, len(audio_arr) / st.session_state.librosa_sr,
#                                                    num=len(audio_arr), dtype=np.float32))  # Time column
#
#     else:
#         wav_df = pd.DataFrame(audio_arr.T, columns=[f"dim_{i}" for i in range(params.nchannels)])
#         wav_df.insert(0, "time_steps", np.linspace(0, wav_df.shape[0] / st.session_state.librosa_sr,
#                                                    num=wav_df.shape[0], dtype=np.float32))
#     wav_df = wav_df.reset_index(drop=True)
#     return wav_df


def _wav_array_to_wav_dataframe(audio_arr):
    """
    Converts audio array read with librosa, mono-channel to dataframe with integrated time steps in seconds.
    :param audio_arr: Audio array read with librosa
    :return: wav_df - audio DataFrame with columns: time_steps, dim_0
    """
    wav_df = pd.DataFrame(audio_arr, columns=[f"dim_0"])  # Audio data
    wav_df.insert(0, "time_steps", np.linspace(0, len(audio_arr) / st.session_state.librosa_sr,
                                               num=len(audio_arr), dtype=np.float32))  # Time column
    wav_df = wav_df.reset_index(drop=True)
    return wav_df


def load_wav_st_uploader(wav_uploader):
    """ Creates a list of Dataframes from streamlit file_uploader and adds dataframes as well as
        other information to streamlit.session_state.

        :param wav_uploader: A list with the uploaded .wav files
        :type wav_uploader: list
        """
    if len(wav_uploader) > 0:
        wav_file_names = []
        wav_names_dict = {}
        wav_index_dict = {}
        wav_dict_index = 0

        wav_parameters = []
        wav_df_list = []

        # Get and sort out names
        file_names = [file.name for file in wav_uploader]
        sorted_file_names = sorted(file_names)

        for name in sorted_file_names:
            for uploaded_file in wav_uploader:
                if uploaded_file.name == name:
                    # Initiate progress bar
                    progress_bar = st.progress(0, text=name)
                    wav_file_names.append(name)
                    wav_names_dict[name] = wav_dict_index  # {"Name.wav": 0}
                    wav_index_dict[wav_dict_index] = name  # {0: "Name.wav"}
                    wav_dict_index = wav_dict_index + 1

                    wav_arr, params = _open_wav(uploaded_file)
                    wav_parameters.append(params)
                    wav_df = _wav_array_to_wav_dataframe(wav_arr)
                    wav_df_list.append(wav_df)
                    progress_bar.progress(100, text=name + " successful")

        st.session_state.wav_file_names = wav_file_names
        st.session_state.wav_names_dict = wav_names_dict
        st.session_state.wav_index_dict = wav_index_dict
        st.session_state.wav_parameters = wav_parameters
        st.session_state.wav_dataframes = wav_df_list


def load_wav_st_multiselect(wav_uploader):
    """ Creates a list of Dataframes from a given local path and adds dataframes as well as
        other information (file_names, parameters, index_dict, names_dict) to streamlit.session_state.

        :param wav_uploader: A list with the uploaded .wav files
        :type wav_uploader: list
        """
    if len(wav_uploader) > 0:
        wav_file_names = []
        wav_names_dict = {}
        wav_index_dict = {}
        wav_dict_index = 0

        wav_parameters = []
        wav_df_list = []

        # Get and sort all file names
        file_names = [file for file in wav_uploader]
        sorted_file_names = sorted(file_names)

        # Case of all files option chosen
        if "All Files" in sorted_file_names:
            # Initiate one progress bar for all
            percent_complete = 0
            all_files_bar = st.progress(percent_complete, text="Loading all files in the dataset... ")

            for name in sorted(os.listdir('./ICBHI_dataset/audio_and_txt')):
                if name.endswith(".wav"):
                    all_files_bar.progress(percent_complete, text=f"Loading all files in the dataset... {name}")

                    wav_path = os.getcwd() + os.sep + "ICBHI_dataset" + os.sep + "audio_and_txt" + os.sep + name
                    # save name and name-index dictionaries
                    wav_file_names.append(name)
                    wav_names_dict[name] = wav_dict_index  # {"Name.wav": 0}
                    wav_index_dict[wav_dict_index] = name  # {0: "Name.wav"}
                    wav_dict_index = wav_dict_index + 1

                    wav_arr, params = _open_wav(wav_path)
                    wav_parameters.append(params)

                    wav_df = _wav_array_to_wav_dataframe(wav_arr)
                    wav_df_list.append(wav_df)

                    # Add to progress bar and avoid reaching 1.0
                    percent_complete = percent_complete + 1 / 920
                    if percent_complete >= 1.0:
                        percent_complete = 0.999

            all_files_bar.progress(1.0, text="Uploading complete.")

        else:  # Users have not chosen "All Files", but a random list of files.
            for name in sorted_file_names:
                progress_bar = st.progress(0, text=name)

                wav_path = os.getcwd() + os.sep + "ICBHI_dataset" + os.sep + "audio_and_txt" + os.sep + name
                # save name and name-index dictionaries
                wav_file_names.append(name)
                wav_names_dict[name] = wav_dict_index  # {"Name.wav": 0}
                wav_index_dict[wav_dict_index] = name  # {0: "Name.wav"}
                wav_dict_index = wav_dict_index + 1

                wav_arr, params = _open_wav(wav_path)
                wav_parameters.append(params)

                wav_df = _wav_array_to_wav_dataframe(wav_arr)
                wav_df_list.append(wav_df)
                progress_bar.progress(100, text=name + " successful")

        st.session_state.wav_file_names = wav_file_names
        st.session_state.wav_names_dict = wav_names_dict
        st.session_state.wav_index_dict = wav_index_dict
        st.session_state.wav_parameters = wav_parameters
        st.session_state.wav_dataframes = wav_df_list
