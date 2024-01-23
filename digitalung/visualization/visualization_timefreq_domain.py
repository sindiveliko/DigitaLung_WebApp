"""
Collection of Functions used to plot Mel Spectrogram of wav-files.
"""

from copy import deepcopy

import librosa
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import streamlit as st

from general_utilities import utilities


@st.cache_resource(show_spinner=False)
def wav_specgram_single_file_detail(wav_df, name, index, rec_info_state, pat_state, az_state, f_min, f_max):
    """
    Plot the mel spectrogram of audio file, accompanied by metadata upon user request.
    :param wav_df: Audio file in dataframe format
    :param name: Name of audio file
    :param index: Audio original index to extract information
    :param rec_info_state: checkbox output - display recording information
    :param pat_state: checkbox output - display clinical information
    :param az_state: checkbox output - display respiratory cycles information
    :param f_min: numerical output - minimum frequency to display in y-axis
    :param f_max: numerical output - maximum frequency to display in y-axis
    """

    # Prep data
    audio_df = deepcopy(wav_df)
    y = np.asarray(audio_df.iloc[:, 1])
    sr = st.session_state.visualization_sr
    # Choose x-axis ticks based on duration
    if (len(y)/sr) >= 60:
        major_tick, minor_tick = 5, 1
    else:
        major_tick, minor_tick = 2, 0.5

    # Audio widget
    st.markdown('---')
    original_data = st.session_state.wav_dataframes[index].iloc[:, 1]
    st.audio(np.array(original_data), sample_rate=st.session_state.librosa_sr)

    # Main figure
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 8), height_ratios=[1, 0.7, 3, 1],
                             width_ratios=[20, 1], squeeze=True)

    # Waveform subplot
    axes[1, 0].plot(np.linspace(0, len(y) - 1, num=len(y)), y, linewidth=0.4, color='black')
    axes[1, 0].set_xlim(0, len(y) - 1)
    axes[1, 0].set_ylim(min(y), max(y))
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    axes[1, 0].set_xticklabels([])
    axes[1, 0].set_yticklabels([])

    # Mel spectrogram
    bbox_props = dict(facecolor='white', edgecolor='black', linewidth=1, alpha=0.7)
    axes[0, 0].set_title(name, y=1.1, x=0.5, fontsize=10, bbox=bbox_props)

    W = 512
    N = W*2
    H = int(0.125 * W)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, win_length=W, hop_length=H, n_fft=N,
                                              window='hann', n_mels=256)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    img = librosa.display.specshow(mel_db, sr=sr, y_axis='mel', x_axis='time', win_length=W, hop_length=H, n_fft=N,
                                   cmap='magma', ax=axes[2, 0])

    axes[2, 0].set_ylabel('Frequency [Hz]')
    axes[2, 0].set_xlabel('Time [s]')
    axes[2, 0].set_ylim([f_min, f_max])
    axes[2, 0].xaxis.set_major_locator(MultipleLocator(major_tick))
    axes[2, 0].xaxis.set_minor_locator(MultipleLocator(minor_tick))

    cbar = fig.colorbar(img, cax=axes[2, 1], format="%+2.f dB")
    cbar.ax.tick_params(labelsize=8)

    # Respiratory cycles as annotations in the figure
    if az_state is True:
        az_df = st.session_state.wav_az_dict[index]
        text_map = {
            (0, 0): 'N',
            (1, 0): 'C',
            (0, 1): 'W',
            (1, 1): 'B'
        }
        for i, row in az_df.iterrows():
            start = row['start']
            end = row['end']
            crackles = row['crackles']
            wheezes = row['wheezes']
            txt = text_map[(crackles, wheezes)]

            axes[2, 0].axvline(x=end, color='white', linestyle='dashed', linewidth=1)

            bbox_props_az = dict()
            if txt == 'N':
                bbox_props_az = dict(boxstyle='round', facecolor='lightgreen',
                                     edgecolor='black', linewidth=1, alpha=0.7)
            elif txt == 'C':
                bbox_props_az = dict(boxstyle='round', facecolor='yellow', edgecolor='black', linewidth=1,
                                     alpha=0.7)
            elif txt == 'W':
                bbox_props_az = dict(boxstyle='round', facecolor='orange', edgecolor='black', linewidth=1,
                                     alpha=0.7)
            elif txt == 'B':
                bbox_props_az = dict(boxstyle='round', facecolor='red', edgecolor='black', linewidth=1,
                                     alpha=0.7)
            x_pos = start + (end - start) / 2
            y_pos = f_max
            axes[2, 0].text(x_pos, y_pos, txt, ha='center', fontsize=10, weight='bold', bbox=bbox_props_az)

        first_row = az_df.iloc[0]
        start_0 = first_row['start']
        axes[2, 0].axvline(x=start_0, color='white', linestyle='dashed', linewidth=1)

        first_column = az_df.iloc[:, 0]
        last_element = az_df.iloc[-1, 1]
        first_column.loc[len(first_column)] = last_element
        x_ticks_v = first_column

        first_column_r = first_column.apply(utilities.round_up_f_dec)  # round up to first decimal
        x_ticks_n = first_column_r

        axes[2, 0].set_xticks(x_ticks_v)
        axes[2, 0].set_xticklabels(x_ticks_n.tolist(), rotation=55)

    # Recording information as table
    rec_index = '---'
    rec_loc = '---'
    acq_m = '---'
    mic = '---'

    if rec_info_state is True:
        rec_index = st.session_state.wav_rec_dict[index]['recording_index']
        rec_loc = st.session_state.wav_rec_dict[index]['chest_location']
        acq_m = st.session_state.wav_rec_dict[index]['acquisition_mode']
        mic = st.session_state.wav_rec_dict[index]['recording_equipment']

    rec_data = [
        ['Recording Index', 'Recording Location', 'Channels', 'Equipment'],
        [rec_index, rec_loc, acq_m, mic]
    ]
    table_rec = axes[0, 0].table(cellText=rec_data, loc='center', cellLoc='center')
    axes[0, 0].axis('off')
    table_rec.auto_set_font_size(False)
    table_rec.set_fontsize(10)
    table_rec.scale(1, 2)
    for i in range(0, len(rec_data)):
        for j in range(0, len(rec_data[0]) - 1):
            cell = table_rec.get_celld()[i, j]
            cell.set_width(0.2)
    for i in range(0, len(rec_data)):
        cell = table_rec.get_celld()[i, 3]
        cell.set_width(0.4)

    # Patient information as table
    pat_id = '---'
    diagnose = '---'
    age = '---'
    sex = '---'
    adult_bmi = '---'
    ch_weight = '---'
    ch_height = '---'

    if pat_state is True:
        pat_id = st.session_state.wav_rec_dict[index]['patient_number']
        diagnose_df = st.session_state.wav_diagnose_df
        diagnose = diagnose_df.loc[diagnose_df['patient_id'] == int(pat_id), 'diagnosis'].iloc[0]
        age, sex, adult_bmi, ch_weight, ch_height = utilities.__get_demog_info(pat_id)
        if np.isnan(adult_bmi):
            adult_bmi = '---'
        else:
            ch_height = '---'
            ch_weight = '---'
    pat_data = [
        ['Patient-ID', 'Diagnose', 'Age (years)', 'Gender', 'Adult BMI (kg/m2)', 'Child-Weight (kg)',
         'Child-Height (cm)'],
        [pat_id, diagnose, age, sex, adult_bmi, ch_weight, ch_height]
    ]
    table_pat = axes[3, 0].table(cellText=pat_data, loc='center', cellLoc='center')
    axes[3, 0].axis('off')
    table_pat.auto_set_font_size(False)
    table_pat.set_fontsize(10)
    table_pat.scale(1, 2)

    # Plot and layout
    plt.subplots_adjust(hspace=0.01)
    plt.tight_layout()
    axes[0, 1].axis('off')
    axes[1, 1].axis('off')
    axes[3, 1].axis('off')
    st.pyplot(fig)
    st.markdown('---')


@st.cache_resource(show_spinner=False)
def wav_specgram_all_files(file_names_list, df_list, az_state):
    """
    Plot mel spectrogram of all audio files, one figure per file, two figures per row.
    :param file_names_list: List of file names
    :param df_list:  List of audio files in dataframe format
    :param az_state: Checkbox output - display respiratory cycles information
    """
    # Iterate over file names two at a time
    for i in range(0, len(file_names_list), 2):
        cols = st.columns(2)

        # Audio output per each two files at a time
        for j in range(2):
            if i + j < len(file_names_list):
                original_data = st.session_state.wav_dataframes[i + j].iloc[:, 1]
                cols[j].audio(np.array(original_data), sample_rate=st.session_state.librosa_sr)

        # Create a large figure to accommodate both columns
        fig = plt.figure(figsize=(16, 6), layout='constrained')

        # Outer Specification
        gs0 = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

        for j in range(2):
            gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, width_ratios=[20, 1], height_ratios=[1, 3.5],
                                                    wspace=0.01, subplot_spec=gs0[j])

            # This handles the case where there's an odd number of files. The second column is blank.
            if i + j >= len(file_names_list):
                ax0 = fig.add_subplot(gs00[0, 0])
                ax1 = fig.add_subplot(gs00[1, 0])
                ax2 = fig.add_subplot(gs00[1, 1])
                ax0.axis('off')
                ax1.axis('off')
                ax2.axis('off')
                break

            name = file_names_list[i + j]
            wav_df = df_list[i + j]
            audio_df = deepcopy(wav_df)
            data = audio_df.iloc[:, 1]
            y = np.array(data)
            sr = st.session_state.visualization_sr

            if (len(y) / sr) >= 60:
                major_tick, minor_tick = 5, 1
            else:
                major_tick, minor_tick = 2, 0.5

            # Waveform
            ax0 = fig.add_subplot(gs00[0, 0])
            ax0.plot(np.linspace(0, len(y) - 1, num=len(y)), y, linewidth=0.4, color='black')
            ax0.set_xlim(0, len(y) - 1)
            ax0.set_ylim(min(y), max(y))
            ax0.set_xticks([])
            ax0.set_yticks([])
            ax0.set_xticklabels([])
            ax0.set_yticklabels([])
            bbox_props = dict(facecolor='white', edgecolor='black', linewidth=1, alpha=0.7)
            ax0.set_title(name, y=1.1, x=0.5, fontsize=12, bbox=bbox_props)

            # Mel spectrogram
            ax1 = fig.add_subplot(gs00[1, 0])
            W = 512
            N = W*2
            H = int(0.125 * W)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, win_length=W, hop_length=H, n_fft=N)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            img = librosa.display.specshow(mel_db, sr=sr, y_axis='mel', x_axis='s', win_length=W, hop_length=H,
                                           n_fft=N, cmap='magma', ax=ax1)
            ax1.set_ylabel('Frequency [Hz]', fontdict=dict(fontsize=10))
            ax1.set_xlabel('Time [s]', fontdict=dict(fontsize=10))
            ax1.tick_params(axis='x', rotation=0, labelsize=10)
            ax1.tick_params(axis='y', rotation=0, labelsize=10)
            ax1.xaxis.set_major_locator(MultipleLocator(major_tick))
            ax1.xaxis.set_minor_locator(MultipleLocator(minor_tick))

            # Respiratory cycles
            if az_state is True:
                az_df = st.session_state.wav_az_dict[i + j]
                text_map = {
                    (0, 0): 'N',
                    (1, 0): 'C',
                    (0, 1): 'W',
                    (1, 1): 'B'
                }
                for k, row in az_df.iterrows():
                    start = row['start']
                    end = row['end']
                    crackles = row['crackles']
                    wheezes = row['wheezes']
                    txt = text_map[(crackles, wheezes)]

                    ax1.axvline(x=end, color='white', linestyle='dashed', linewidth=1)

                    bbox_props_az = dict()
                    if txt == 'N':
                        bbox_props_az = dict(boxstyle='round', facecolor='lightgreen',
                                             edgecolor='black', linewidth=1, alpha=0.7)
                    elif txt == 'C':
                        bbox_props_az = dict(boxstyle='round', facecolor='yellow', edgecolor='black', linewidth=1,
                                             alpha=0.7)
                    elif txt == 'W':
                        bbox_props_az = dict(boxstyle='round', facecolor='orange', edgecolor='black', linewidth=1,
                                             alpha=0.7)
                    elif txt == 'B':
                        bbox_props_az = dict(boxstyle='round', facecolor='red', edgecolor='black', linewidth=1,
                                             alpha=0.7)
                    x_pos = start + (end - start) / 2
                    y_pos = sr / 2
                    ax1.text(x_pos, y_pos, txt, ha='center', fontsize=6, weight='bold', bbox=bbox_props_az)

                first_row = az_df.iloc[0]
                start_0 = first_row['start']
                ax1.axvline(x=start_0, color='white', linestyle='dashed', linewidth=1)

                first_column = az_df.iloc[:, 0]
                last_element = az_df.iloc[-1, 1]
                first_column.loc[len(first_column)] = last_element

                x_ticks_v = first_column
                x_ticks_n = first_column.apply(utilities.round_up_f_dec)  # round up to first decimal

                ax1.set_xticks(x_ticks_v)
                ax1.set_xticklabels(x_ticks_n.tolist(), fontsize=6, rotation=55)

            # Plot color-bar
            ax2 = fig.add_subplot(gs00[1, 1])
            cbar = fig.colorbar(img, cax=ax2, format="%+2.f dB")
            cbar.ax.tick_params(labelsize=10)

        st.pyplot(fig, use_container_width=True)
        st.markdown('---')


@st.cache_resource(show_spinner=False)
def wav_specgram_multiple_files(selectbox, file_names_list, df_list, az_state):
    """
    Plot mel spectrogram of multiple audio files, selected from the user, one figure per file, two figures per row.
    :param selectbox: User selection of files obtained with st.selectbox
    :param file_names_list: Names corresponding to each file
    :param df_list:  Audios in dataframe format
    :param az_state: Checkbox output - Respiratory cycles information
    """
    # Iterate over file names two at a time
    for i in range(0, len(selectbox), 2):
        # Display audio
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(selectbox):
                name = selectbox[i + j]
                index = file_names_list.index(name)
                original_data = st.session_state.wav_dataframes[index].iloc[:, 1]
                cols[j].audio(np.array(original_data), sample_rate=st.session_state.librosa_sr)

        # Create a large figure to accommodate both columns
        fig = plt.figure(figsize=(16, 6), layout='constrained')
        gs0 = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

        for j in range(2):
            gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, width_ratios=[20, 1], height_ratios=[1, 3.5],
                                                    wspace=0.01, subplot_spec=gs0[j])
            if i + j >= len(selectbox):  # This handles the case where there's an odd number of files.
                ax0 = fig.add_subplot(gs00[0, 0])
                ax1 = fig.add_subplot(gs00[1, 0])
                ax2 = fig.add_subplot(gs00[1, 1])
                ax0.axis('off')
                ax1.axis('off')
                ax2.axis('off')
                break

            name = selectbox[i + j]
            index = file_names_list.index(name)
            wav_df = df_list[index]
            audio_df = deepcopy(wav_df)
            data = audio_df.iloc[:, 1]
            y = np.array(data)
            sr = st.session_state.visualization_sr

            if (len(y) / sr) >= 60:
                major_tick, minor_tick = 5, 1
            else:
                major_tick, minor_tick = 2, 0.5

            # Waveform
            ax0 = fig.add_subplot(gs00[0, 0])
            ax0.plot(np.linspace(0, len(y) - 1, num=len(y)), y, linewidth=0.4, color='black')
            ax0.set_xlim(0, len(y) - 1)
            ax0.set_ylim(min(y), max(y))
            ax0.set_xticks([])
            ax0.set_yticks([])
            ax0.set_xticklabels([])
            ax0.set_yticklabels([])
            bbox_props = dict(facecolor='white', edgecolor='black', linewidth=1, alpha=0.7)
            ax0.set_title(name, y=1.1, x=0.5, fontsize=12, bbox=bbox_props)

            # Mel spectrogram
            ax1 = fig.add_subplot(gs00[1, 0])
            W = 512
            N = W * 2
            H = int(0.125 * W)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, win_length=W, hop_length=H, n_fft=N)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            img = librosa.display.specshow(mel_db, sr=sr, y_axis='mel', x_axis='s', win_length=W, hop_length=H, n_fft=N,
                                           cmap='magma', ax=ax1)
            ax1.set_ylabel('Frequency [Hz]', fontdict=dict(fontsize=10))
            ax1.set_xlabel('Time [s]', fontdict=dict(fontsize=10))
            ax1.tick_params(axis='x', rotation=0, labelsize=10)
            ax1.tick_params(axis='y', rotation=0, labelsize=10)
            ax1.xaxis.set_major_locator(MultipleLocator(major_tick))
            ax1.xaxis.set_minor_locator(MultipleLocator(minor_tick))

            # Respiratory cycles
            if az_state is True:
                az_df = st.session_state.wav_az_dict[index]
                text_map = {
                    (0, 0): 'N',
                    (1, 0): 'C',
                    (0, 1): 'W',
                    (1, 1): 'B'
                }
                for k, row in az_df.iterrows():
                    start = row['start']
                    end = row['end']
                    crackles = row['crackles']
                    wheezes = row['wheezes']
                    txt = text_map[(crackles, wheezes)]

                    ax1.axvline(x=end, color='white', linestyle='dashed', linewidth=1)

                    bbox_props_az = dict()
                    if txt == 'N':
                        bbox_props_az = dict(boxstyle='round', facecolor='lightgreen',
                                             edgecolor='black', linewidth=1, alpha=0.7)
                    elif txt == 'C':
                        bbox_props_az = dict(boxstyle='round', facecolor='yellow', edgecolor='black', linewidth=1,
                                             alpha=0.7)
                    elif txt == 'W':
                        bbox_props_az = dict(boxstyle='round', facecolor='orange', edgecolor='black', linewidth=1,
                                             alpha=0.7)
                    elif txt == 'B':
                        bbox_props_az = dict(boxstyle='round', facecolor='red', edgecolor='black', linewidth=1,
                                             alpha=0.7)
                    x_pos = start + (end - start) / 2
                    y_pos = sr / 2
                    ax1.text(x_pos, y_pos, txt, ha='center', fontsize=6, weight='bold', bbox=bbox_props_az)

                first_row = az_df.iloc[0]
                start_0 = first_row['start']
                ax1.axvline(x=start_0, color='white', linestyle='dashed', linewidth=1)

                first_column = az_df.iloc[:, 0]
                last_element = az_df.iloc[-1, 1]
                first_column.loc[len(first_column)] = last_element

                x_ticks_v = first_column
                x_ticks_n = first_column.apply(utilities.round_up_f_dec)  # round up to first decimal

                ax1.set_xticks(x_ticks_v)
                ax1.set_xticklabels(x_ticks_n.tolist(), fontsize=6, rotation=55)

            # Plot color bar
            ax2 = fig.add_subplot(gs00[1, 1])
            cbar = fig.colorbar(img, cax=ax2, format="%+2.f dB")
            cbar.ax.tick_params(labelsize=10)

        st.pyplot(fig, use_container_width=True)
        st.markdown('---')
