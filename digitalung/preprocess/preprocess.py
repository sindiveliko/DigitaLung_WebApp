"""Collection of Functions used in the page for preprocessing.
It contains peak normalization and channel analysis.
"""

import librosa
import numpy as np
import pandas as pd
import streamlit as st


def audio_scaling():
    """This functions performs peak normalization for all the dataframes (corresponding to wav-files) in session-state.
    The new values range from -1 to +1.

    :return: A list of normalized dataframes corresponding to WAV files
    :rtype: list
    """
    percent_complete = 0
    progress_bar = st.progress(percent_complete, text="Applying audio scaling to audio files...")

    # Peak Normalization - Audio Scaling
    df_audio_scaling = []
    for i, df in enumerate(st.session_state.wav_dataframes):
        ampl = df.iloc[:, 1].values
        # Downsample as a prepping step for visualization
        ampl_col = librosa.resample(ampl, orig_sr=st.session_state.librosa_sr,
                                    target_sr=st.session_state.visualization_sr, res_type='soxr_hq')
        # Audio scaling
        ampl_col = librosa.util.normalize(ampl_col, norm=np.inf)
        # New time column
        time_col = np.linspace(0, len(ampl_col) / st.session_state.visualization_sr, num=len(ampl_col))
        scaled_df = pd.DataFrame({'time_steps': time_col, 'dim_0': ampl_col})
        df_audio_scaling.append(scaled_df)

        percent_complete = percent_complete + 1/len(st.session_state.wav_dataframes)
        if percent_complete >= 1.0:
            percent_complete = 0.99
        progress_bar.progress(percent_complete, text="Applying audio scaling to audio files...")

    progress_bar.empty()
    return df_audio_scaling


# Could be potentially interesting for future work to work with multichannel files.
# Check up the format and add accordingly by visualizing channels, down-mixing, or splitting them.
# This is the first step of check up.
# def wav_check_channels():
#     """This functions checks all the wav files saved in session state for the number of channels.
#     It returns the answer for channel analysis and the number of multichannel files.
#     :return: Answer "Mono" or "Multi" and number of files in multichannel format.
#     :rtype:Str
#     """
#     answer = 'Mono'
#     multi = 0
#     for df in st.session_state.wav_dataframes:
#         n_channels = df.shape[1]    # col1:time_steps , col2:amplitude
#         if n_channels > 2:
#             answer = 'Multi'
#             multi = multi + 1
#     return answer, multi
