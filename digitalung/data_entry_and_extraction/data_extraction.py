"""
This module contains a collection of functions that are used to upload all the metadata available
related to the ICBHI Dataset.

- Recording Information extracted from file name per patient
- Respiratory Information uploaded per each patient (cycles and labels)
- Diagnose Information uploaded for all patients
- Demographic Information uploaded for all patients
"""

import os

import pandas as pd
import streamlit as st


def extract_patient_information():
    """
    Extract and save to session state: Recording, Respiratory Cycles and Demographic Information, and Diagnose.
    """
    progress_bar = st.progress(0, text='Recording Information in progress')
    st.session_state.wav_rec_dict = _wav_get_rec_info()
    progress_bar.progress(100, text='Recording Information successfully saved')
    progress_bar = st.progress(0, text='Demographic Information in progress')
    st.session_state.wav_demog_df = _wav_get_demog_info()
    progress_bar.progress(100, text='Demographic Information successfully saved')
    progress_bar = st.progress(0, text='Diagnose in progress')
    st.session_state.wav_diagnose_df = _wav_get_diagnose()
    progress_bar.progress(100, text='Diagnose successfully saved')
    progress_bar = st.progress(0, text='Respiratory Cycles in progress')
    st.session_state.wav_az_dict = _wav_get_resp_info()
    progress_bar.progress(100, text='Respiratory Cycles successfully saved')


def _wav_get_rec_info():
    """ Creates and returns a Dictionary containing the recording information of all WAV-files saved in session state.
    The elements of the file name are mapped as recording information.

            :return: Dictionary with recording information
            :rtype:dict
            """
    rec_dict = {}
    for i, file_name in enumerate(st.session_state.wav_file_names):
        elements = file_name.split("_")
        # Extracting individual elements from the file name
        patient_id = elements[0]
        recording_index = elements[1]
        chest_location = elements[2]
        acquisition_mode = elements[3]
        recording_equipment = elements[4].split(".")[0]  # Remove the file extension

        # Mapping chest location codes to their full names
        chest_location_mapping = {
            "Al": "Anterior Left",
            "Ar": "Anterior Right",
            "Ll": "Lateral Left",
            "Lr": "Lateral Right",
            "Pl": "Posterior Left",
            "Pr": "Posterior Right",
            "Tc": "Trachea",
        }
        chest_location_full = chest_location_mapping.get(chest_location, chest_location)

        # Mapping acquisition mode codes to their full names
        acquisition_mode_mapping = {
            "sc": "Single",
            "mc": "Multi"
        }
        acquisition_mode_full = acquisition_mode_mapping.get(acquisition_mode, acquisition_mode)

        # ## Optional: the name can be too long to display.
        # # Mapping recording equipment codes to their full names

        # ## Use full names
        # recording_equipment_mapping = {
        #     "AKGC417L": "AKG C417L Microphone",
        #     "LittC2SE": "3M Littmann Classic II SE Stethoscope",
        #     "Litt3200": "3M Littmann 3200 Electronic Stethoscope",
        #     "Meditron": "WelchAllyn Meditron Electronic Stethoscope"
        # }
        # ## or continue with abbreviation
        recording_equipment_mapping = {
            "AKGC417L": "AKGC417L",
            "LittC2SE": "LittC2SE",
            "Litt3200": "Litt3200",
            "Meditron": "Meditron"
        }
        recording_equipment_full = recording_equipment_mapping.get(recording_equipment, recording_equipment)

        rec_dict[i] = {
            "file_name": file_name,
            "patient_number": patient_id,
            "recording_index": recording_index,
            "chest_location": chest_location_full,
            "acquisition_mode": acquisition_mode_full,
            "recording_equipment": recording_equipment_full
        }
    return rec_dict


def _wav_get_resp_info():
    """ Create a dictionary containing key: file_name.wav_lib and value: Dataframe with information about respiratory
    cycles.

       :return: Dataframe containing respiratory cycle information
       :rtype: dict
       """
    az_dict = {}
    codings = []
    for i, file in enumerate(st.session_state.wav_file_names):
        name = file.split(".")[0] + ".txt"
        az_dir_path = './ICBHI_dataset/audio_and_txt'
        for filename in os.listdir(az_dir_path):
            if name == filename:
                az_path = az_dir_path + os.sep + name
                az_df = pd.read_csv(az_path, delimiter='	', names=['start', 'end', 'crackles', 'wheezes'])
                az_dict[i] = az_df

                coding_map = {
                    (0, 0): 'N',
                    (1, 0): 'C',
                    (0, 1): 'W',
                    (1, 1): 'CW'
                }
                coding = ''
                for j, r in az_df.iterrows():
                    crackles = r['crackles']
                    wheezes = r['wheezes']
                    coding = coding + coding_map[(crackles, wheezes)] + " "
                codings.append(coding)  # a list of letters representing cycles coding on that file [N,C,B,C,W,N...]

    st.session_state.wav_codings = codings
    os.chdir(st.session_state.base_dir)
    return az_dict


def _wav_get_diagnose():
    """ Create a dataframe from the csv file containing the patient-id and diagnose for all patients.

    :return: Dataframe containing diagnoses
    :rtype:pd.DataFrame
    """
    diagnose_path = './ICBHI_dataset/audio_diagnose/patient_diagnosis.csv'
    diagnose_df = pd.read_csv(diagnose_path, header=None, names=['patient_id', 'diagnosis'])
    os.chdir(st.session_state.base_dir)
    return diagnose_df


def _wav_get_demog_info():
    """ Create a dataframe from the txt file containing the demographic information for all patients.

       :return: Dataframe containing demographic information
       :rtype:pd.DataFrame
       """
    demog_path = './ICBHI_dataset/demographic_info.txt'
    demog_df = pd.read_csv(demog_path, delimiter=' ',
                           names=['patient_id', 'age', 'sex', 'adult_bmi', 'ch_weight', 'ch_height'])
    os.chdir(st.session_state.base_dir)
    return demog_df


def organize_all_patient_info():
    """
    Organize all extracted patient information to a common pandas DataFrame.

    :return: DataFrame with ["File Name", "Rec.Index", "Chest Location", "Acquisition", "Rec.Equipment",
    "Patient-ID", "Age (years)", "Sex", "Adult BMI (kg/m^2)", "Child Weight (kg)", "Child Height (cm)", "Diagnosis",
    "Nr. of respiratory cycles", "Average cycle duration (s)", "Nr. of Normal", "Nr. of Crackle", "Nr. of Wheeze",
    "Nr. of Both"]
    """
    # Convert dictionaries to DataFrames
    recording_df = pd.DataFrame.from_dict(st.session_state.wav_rec_dict, orient='index').astype(str)
    demographic_df = st.session_state.wav_demog_df.astype(str)
    diagnosis_df = st.session_state.wav_diagnose_df.astype(str)

    # Extract information from respiratory dictionary
    nr_of_cycles, average_cycle_duration = [], []
    nr_of_n, nr_of_c, nr_of_w, nr_of_b = [], [], [], []

    for idx, (key, temp_df) in enumerate(st.session_state.wav_az_dict.items()):
        nr_of_cycles.append(len(temp_df))
        average_cycle_duration.append((temp_df['end'] - temp_df['start']).mean())

        nr_of_n.append(len(temp_df[(temp_df['crackles'] == 0) & (temp_df['wheezes'] == 0)]))
        nr_of_c.append(len(temp_df[(temp_df['crackles'] == 1) & (temp_df['wheezes'] == 0)]))
        nr_of_w.append(len(temp_df[(temp_df['crackles'] == 0) & (temp_df['wheezes'] == 1)]))
        nr_of_b.append(len(temp_df[(temp_df['crackles'] == 1) & (temp_df['wheezes'] == 1)]))

    az_result_base_data = {
        'nr_of_cycles': nr_of_cycles,
        'average_cycle_duration': average_cycle_duration,
        'nr_of_N': nr_of_n,
        'nr_of_C': nr_of_c,
        'nr_of_W': nr_of_w,
        'nr_of_B': nr_of_b
    }
    respiratory_df = pd.DataFrame(az_result_base_data)

    # Merge DataFrames
    merged_df = pd.merge(recording_df, demographic_df, left_on='patient_number', right_on='patient_id', how='left')
    merged_df = merged_df.drop(['patient_number'], axis=1)
    merged_df = pd.merge(merged_df, diagnosis_df, on='patient_id', how='left')
    merged_df = pd.merge(merged_df, respiratory_df, left_index=True, right_index=True, how='left')
    merged_df.columns = ["File Name", "Rec. Index", "Chest Location", "Acquisition", "Rec. Equipment",
                         "Patient-ID", "Age (years)", "Sex", "Adult BMI (kg/m^2)", "Child Weight (kg)",
                         "Child Height (cm)", "Diagnosis", "Nr. of resp. cycles", "Average cycle duration (s)",
                         "Nr. of Normal", "Nr. of Crackle", "Nr. of Wheeze", "Nr. of Both"]

    return merged_df
