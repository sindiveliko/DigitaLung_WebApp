"""
This module contains a collection of functions that are used to create the questionnaire regarding patient data entry.
"""

import base64

import numpy as np
import streamlit as st


def questionnaire():
    """
    Displays the questionnaire in a form.
    Saves information in session state.
    Offers a .csv file to download patient information for future uses.
    """
    with st.form('Digital Data Entry'):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('### Demographic Data')
            st.markdown("Patient pseudonym: Is assigned automatically by the software")
            patient_pseudonym = st.text_input("Pseudonym - 'Patient pseudonym'", "")
            patient_surname = st.text_input("Surname – 'Patient surname'", "")
            patient_name = st.text_input("Name – 'Patient name'", "")
            patient_date_of_birth = st.text_input("Date of birth - 'in dd.mm.yyyy'", "")
            patient_age = st.text_input("Age - 'as integer'")
            patient_gender = st.selectbox('Gender', ('unknown', 'male', 'female', 'diverse'))
            patient_consent = st.selectbox("Patient Consent", ("yes", "no"))

        with col2:
            st.markdown('### Clinical Data and Medical History')
            # """Parameters:
            # Height - in cm
            # Weight - in kg
            # BMI – body mass index
            # Pregnancy – true, false
            # Diagnosis (general) - According to selection list: { Chronic bronchitis, COPD (+ GOLD stage),
            # emphysema, asthma (including GINA stage), tuberculosis, interstitial lung disease (pulmonary fibrosis)
            # , post COVID, bronchiectasis, infection of the upper respiratory tract, pneumonia, lung carcinoma }
            # Multiple selection is possible
            # Smoking history – yes, no
            # Current diagnosis - According to a pick list { COPD exacerbation, COPD stable, asthma exacerbation,
            # asthma stable, pulmonary fibrosis exacerbation, pulmonary fibrosis stable, bronchiectasis
            # exacerbation, bronchiectasis stable }
            # Therapies and operations affecting the lungs - According to selection list: { lobectomy,
            # pneumonectomy, radiation of the lungs } Multiple selection is possible
            # blood pressure – multiple measurements (diastolic, systolic, etc.)
            # lung function test
            #
            # """

            patient_height = st.text_input("Height - 'in cm'", 0)
            patient_weight = st.text_input("Weight - 'in kg'", 0)
            patient_pregnancy = st.selectbox("Pregnancy", ("yes", "no"))
            patient_diagnosis_general_multiselect = st.multiselect('Diagnosis (General):',
                                                                   ("Chronic bronchitis",
                                                                    "COPD (+ GOLD stage)",
                                                                    "emphysema",
                                                                    "asthma (including "
                                                                    "GINA stage)",
                                                                    "tuberculosis",
                                                                    "interstitial lung "
                                                                    "disease (pulmonary "
                                                                    "fibrosis)",
                                                                    "post COVID",
                                                                    "bronchiectasis",
                                                                    "infection of the upper"
                                                                    " respiratory tract",
                                                                    "pneumonia",
                                                                    "lung carcinoma"))
            patient_smoking_history = st.selectbox("Smoking History", ("yes", "no", "former"))
            patient_smoking_year_stopped_if_former = st.text_input("If, former smoker, year in which "
                                                                   "stopped smoking", "")
            patient_current_diagnosis = st.selectbox("Current Diagnosis", ("COPD exacerbation",
                                                                           "COPD stable",
                                                                           "asthma exacerbation",
                                                                           "asthma stable",
                                                                           "pulmonary fibrosis exacerbation",
                                                                           "pulmonary fibrosis stable",
                                                                           "bronchiectasis exacerbation",
                                                                           "bronchiectasis stable"))
            patient_date_of_the_current_diagnosis = st.text_input(
                "Date of the current diagnosis - 'in dd.mm.yyyy'",
                "")
            patient_therapies_and_operations_affecting_the_lungs_multiselect = st.multiselect(
                'Therapies and operations affecting the lungs:',
                ("lobectomy", "pneumonectomy",
                 "radiation of the lungs")
            )
            st.write("###### Blood Pressure")
            patient_diastolic_blood_pressure = st.text_input("Diastolic Blood Pressure - 'in mmHg'", "")
            patient_systolic_blood_pressure = st.text_input("Systolic Blood Pressure - 'in mmHg'", "")
            patient_heart_rate = st.text_input("Heart rate - 'in 1/min'")
            st.write("###### Lung Function Test")
            patient_lung_function_FVC_FEV1 = st.selectbox("Lung function FVC, FEV1",
                                                          ("Yes: Done", "No: Not accomplished"))
            patient_lung_function_test_date = st.text_input(
                "If function test carried out: execution date - 'in mm.dd.yyyy'", "")
            patient_lung_function = st.text_input("If function test carried out: Deviation from target in [%]",
                                                  "")

        with col3:
            st.markdown('### Symptom Questionnaire')
            st.write("###### Symptoms")
            # five symptoms: cough, dyspnea, MRC Dyspnea Scale, sputum, Sputum (if yes color selection (clear,
            # white, yellow, green, red) )
            patient_symptom_cough = st.selectbox("Cough", ("yes", "no"))
            patient_symptom_dyspnea = st.selectbox("Dyspnea", ("yes", "no"))
            patient_symptom_MRC_dyspnea_scale = st.text_input(
                "If symptom is dyspnea, then specify its scale as integer", "")
            patient_symptom_sputum = st.selectbox("Sputum", ("yes", "no"))
            patient_symptom_sputum_color = st.selectbox("If sputum is present, then specify its color",
                                                        ("clear", "white", "yellow", "green", "red"))
            st.write("###### Breathing sounds")
            patient_vesicular_breathing_sound = st.selectbox("Vesicular Breathing Sound", ("yes", "no"))
            patient_bronchial_breathing_sound = st.selectbox("Bronchial Breathing Sound", ("yes", "no"))
            patient_decreased_breathing_sound = st.selectbox("Decreased Breathing Sound", ("yes", "no"))
            patient_localization_of_decreased_breathing_sound = st.selectbox(
                "Localization of Decreased Breathing Sound",
                ("right", "left", "upper field", "midfield", "subfield"))
            st.write("##### Background noise")
            st.write("###### Moist (discontinuous)")
            patient_coarse_bubble_rattling_sound = st.selectbox("Coarse-bubble rattling sound", ("yes", "no"))
            patient_localization_of_coarse_bubble_rattling_sound = st.selectbox(
                "Localization of Coarse-bubble rattling sound",
                ("right", "left", "upper field", "midfield", "subfield"))
            patient_fine_bubble_rattling_sound = st.selectbox("Fine-bubble rattling sound", ("yes", "no"))
            patient_localization_of_fine_bubble_rattling_sound = st.selectbox(
                "Localization of Fine-bubble rattling sound",
                ("right", "left", "upper field", "midfield", "subfield"))
            patient_crackling_rattle_sound = st.selectbox("Crackling rattle sound", ("yes", "no"))
            patient_localization_of_crackling_rattle_sound = st.selectbox(
                "Localization of Crackling rattle sound",
                ("right", "left", "upper field",
                 "midfield", "subfield"))
            st.write("###### Dry (continuous)")
            patient_hum = st.selectbox("Hum", ("yes", "no"))
            patient_whistling_wheezing_inspiratory = st.selectbox("Whistling/Wheezing: Inspiratory",
                                                                  ("yes", "no"))
            patient_whistling_wheezing_expiratory = st.selectbox("Whistling/ Wheezing: Expiratory",
                                                                 ("yes", "no"))
            patient_stridor = st.selectbox("Stridor", ("yes", "no"))

        data_entry_and_extraction_submit_button = st.form_submit_button('Submit')
        if data_entry_and_extraction_submit_button:
            # Demographic Data
            st.session_state.dde_patient_pseudonym = patient_pseudonym
            st.session_state.dde_patient_surname = patient_surname
            st.session_state.dde_patient_name = patient_name
            st.session_state.dde_patient_date_of_birth = patient_date_of_birth
            st.session_state.dde_patient_age = patient_age
            st.session_state.dde_patient_gender = patient_gender
            st.session_state.dde_patient_consent = patient_consent

            # Clinical Data and Medical History
            st.session_state.dde_patient_height = patient_height
            st.session_state.dde_patient_weight = patient_weight
            st.session_state.dde_patient_pregnancy = patient_pregnancy
            st.session_state.dde_patient_diagnosis_general_multiselect = patient_diagnosis_general_multiselect
            st.session_state.dde_patient_smoking_history = patient_smoking_history
            st.session_state.dde_patient_smoking_year_stopped_if_former = patient_smoking_year_stopped_if_former
            st.session_state.dde_patient_current_diagnosis = patient_current_diagnosis
            st.session_state.dde_patient_date_of_the_current_diagnosis = patient_date_of_the_current_diagnosis
            st.session_state.dde_patient_therapies_and_operations_affecting_the_lungs_multiselect = \
                patient_therapies_and_operations_affecting_the_lungs_multiselect
            st.session_state.dde_patient_diastolic_blood_pressure = patient_diastolic_blood_pressure
            st.session_state.dde_patient_systolic_blood_pressure = patient_systolic_blood_pressure
            st.session_state.dde_patient_heart_rate = patient_heart_rate
            st.session_state.dde_patient_lung_function_FVC_FEV1 = patient_lung_function_FVC_FEV1
            st.session_state.dde_patient_lung_function_test_date = patient_lung_function_test_date
            st.session_state.dde_patient_lung_function = patient_lung_function

            # Symptom Questionnaire
            st.session_state.dde_patient_symptom_cough = patient_symptom_cough
            st.session_state.dde_patient_symptom_dyspnea = patient_symptom_dyspnea
            st.session_state.dde_patient_symptom_MRC_dyspnea_scale = patient_symptom_MRC_dyspnea_scale
            st.session_state.dde_patient_symptom_sputum = patient_symptom_sputum
            st.session_state.dde_patient_symptom_sputum_color = patient_symptom_sputum_color
            st.session_state.dde_patient_vesicular_breathing_sound = patient_vesicular_breathing_sound
            st.session_state.dde_patient_bronchial_breathing_sound = patient_bronchial_breathing_sound
            st.session_state.dde_patient_decreased_breathing_sound = patient_decreased_breathing_sound
            st.session_state.dde_patient_localization_of_decreased_breathing_sound = \
                patient_localization_of_decreased_breathing_sound
            st.session_state.dde_patient_coarse_bubble_rattling_sound = patient_coarse_bubble_rattling_sound
            st.session_state.dde_patient_localization_of_coarse_bubble_rattling_sound = \
                patient_localization_of_coarse_bubble_rattling_sound
            st.session_state.dde_patient_fine_bubble_rattling_sound = patient_fine_bubble_rattling_sound
            st.session_state.dde_patient_localization_of_fine_bubble_rattling_sound = \
                patient_localization_of_fine_bubble_rattling_sound
            st.session_state.dde_patient_crackling_rattle_sound = patient_crackling_rattle_sound
            st.session_state.dde_patient_localization_of_crackling_rattle_sound = \
                patient_localization_of_crackling_rattle_sound
            st.session_state.dde_patient_hum = patient_hum
            st.session_state.dde_patient_whistling_wheezing_inspiratory = patient_whistling_wheezing_inspiratory
            st.session_state.dde_patient_whistling_wheezing_expiratory = patient_whistling_wheezing_expiratory
            st.session_state.dde_patient_stridor = patient_stridor

            extract_to_csvs(
                # Demographic Data
                st.session_state.dde_patient_pseudonym,
                st.session_state.dde_patient_surname,
                st.session_state.dde_patient_name,
                st.session_state.dde_patient_date_of_birth,
                st.session_state.dde_patient_age,
                st.session_state.dde_patient_gender,
                st.session_state.dde_patient_consent,

                # Clinical Data and Medical History
                st.session_state.dde_patient_height,
                st.session_state.dde_patient_weight,
                st.session_state.dde_patient_pregnancy,
                st.session_state.dde_patient_diagnosis_general_multiselect,
                st.session_state.dde_patient_smoking_history,
                st.session_state.dde_patient_smoking_year_stopped_if_former,
                st.session_state.dde_patient_current_diagnosis,
                st.session_state.dde_patient_date_of_the_current_diagnosis,
                st.session_state.dde_patient_therapies_and_operations_affecting_the_lungs_multiselect,
                st.session_state.dde_patient_diastolic_blood_pressure,
                st.session_state.dde_patient_systolic_blood_pressure,
                st.session_state.dde_patient_heart_rate,
                st.session_state.dde_patient_lung_function_FVC_FEV1,
                st.session_state.dde_patient_lung_function_test_date,
                st.session_state.dde_patient_lung_function,

                # Symptom Questionnaire
                st.session_state.dde_patient_symptom_cough,
                st.session_state.dde_patient_symptom_dyspnea,
                st.session_state.dde_patient_symptom_MRC_dyspnea_scale,
                st.session_state.dde_patient_symptom_sputum,
                st.session_state.dde_patient_symptom_sputum_color,
                st.session_state.dde_patient_vesicular_breathing_sound,
                st.session_state.dde_patient_bronchial_breathing_sound,
                st.session_state.dde_patient_decreased_breathing_sound,
                st.session_state.dde_patient_localization_of_decreased_breathing_sound,
                st.session_state.dde_patient_coarse_bubble_rattling_sound,
                st.session_state.dde_patient_localization_of_coarse_bubble_rattling_sound,
                st.session_state.dde_patient_fine_bubble_rattling_sound,
                st.session_state.dde_patient_localization_of_fine_bubble_rattling_sound,
                st.session_state.dde_patient_crackling_rattle_sound,
                st.session_state.dde_patient_localization_of_crackling_rattle_sound,
                st.session_state.dde_patient_hum,
                st.session_state.dde_patient_whistling_wheezing_inspiratory,
                st.session_state.dde_patient_whistling_wheezing_expiratory,
                st.session_state.dde_patient_stridor,
            )
            st.write('Questionnaire Submitted.')


def extract_to_csvs(
        # Demographic Data
        patient_pseudonym,
        patient_surname,
        patient_name,
        patient_date_of_birth,
        patient_age,
        patient_gender,
        patient_consent,

        # Clinical Data and Medical History
        patient_height,
        patient_weight,
        patient_pregnancy,
        patient_diagnosis_general_multiselect,
        patient_smoking_history,
        patient_smoking_year_stopped_if_former,
        patient_current_diagnosis,
        patient_date_of_the_current_diagnosis,
        patient_therapies_and_operations_affecting_the_lungs_multiselect,
        patient_diastolic_blood_pressure,
        patient_systolic_blood_pressure,
        patient_heart_rate,
        patient_lung_function_FVC_FEV1,
        patient_lung_function_test_date,
        patient_lung_function,

        # Symptom Questionnaire
        patient_symptom_cough,
        patient_symptom_dyspnea,
        patient_symptom_MRC_dyspnea_scale,
        patient_symptom_sputum,
        patient_symptom_sputum_color,
        patient_vesicular_breathing_sound,
        patient_bronchial_breathing_sound,
        patient_decreased_breathing_sound,
        patient_localization_of_decreased_breathing_sound,
        patient_coarse_bubble_rattling_sound,
        patient_localization_of_coarse_bubble_rattling_sound,
        patient_fine_bubble_rattling_sound,
        patient_localization_of_fine_bubble_rattling_sound,
        patient_crackling_rattle_sound,
        patient_localization_of_crackling_rattle_sound,
        patient_hum,
        patient_whistling_wheezing_inspiratory,
        patient_whistling_wheezing_expiratory,
        patient_stridor
):
    patient_bmi = calculate_bmi(patient_height, patient_weight)
    csv_header = "pseudonym,surname,name,date_of_birth,age,gender,consent," \
                 "height,weight,bmi,pregnancy,diagnosis_general," \
                 "smoking history,smoking_year_stopped_if_former,current_diagnosis,date_of_the_current_diagnosis," \
                 "therapies_and_operations_affecting_the_lungs,diastolic_blood_pressure (mmHg)," \
                 "systolic_blood_pressure (mmHg)," \
                 "heart_rate (1/min),lung_function_FVC_FEV1,lung_function_test_date," \
                 "lung_function (Deviation from target in [%])," \
                 "cough,dyspnea,MRC_dyspnea_scale,sputum,sputum_color," \
                 "vesicular_breathing_sound,bronchial_breathing_sound," \
                 "decreased_breathing_sound,localization_of_decreased_breathing_sound,coarse_bubble_rattling_sound," \
                 "localization_of_coarse_bubble_rattling_sound,fine_bubble_rattling_sound," \
                 "localization_of_fine_bubble_rattling_sound," \
                 "crackling_rattle_sound,localization_of_crackling_rattle_sound,hum,whistling_wheezing_inspiratory," \
                 "whistling_wheezing_expiratory,stridor"

    demographic_data = ",".join([patient_pseudonym,
                                 patient_surname,
                                 patient_name,
                                 patient_date_of_birth,
                                 patient_age,
                                 patient_gender,
                                 patient_consent])

    clinical_data_and_medical_history = ",".join([str(patient_height),
                                                  str(patient_weight),
                                                  str(patient_bmi),
                                                  patient_pregnancy,
                                                  str(patient_diagnosis_general_multiselect),
                                                  patient_smoking_history])
    if patient_smoking_history != "former":
        clinical_data_and_medical_history = ",".join([clinical_data_and_medical_history, ""])
    else:
        clinical_data_and_medical_history = ",".join([clinical_data_and_medical_history,
                                                      patient_smoking_year_stopped_if_former])

    clinical_data_and_medical_history = ",".join([clinical_data_and_medical_history,
                                                  patient_current_diagnosis,
                                                  patient_date_of_the_current_diagnosis,
                                                  str(patient_therapies_and_operations_affecting_the_lungs_multiselect),
                                                  patient_diastolic_blood_pressure,
                                                  patient_systolic_blood_pressure,
                                                  patient_heart_rate,
                                                  patient_lung_function_FVC_FEV1])
    if "No" in patient_lung_function_FVC_FEV1:
        clinical_data_and_medical_history = ",".join([clinical_data_and_medical_history, "", ""])
    else:
        clinical_data_and_medical_history = ",".join([clinical_data_and_medical_history,
                                                      patient_lung_function_test_date,
                                                      patient_lung_function])

    symptom_questionnaire = ",".join([patient_symptom_cough,
                                      patient_symptom_dyspnea])

    if patient_symptom_dyspnea == "no":
        symptom_questionnaire = ",".join([symptom_questionnaire, "", patient_symptom_sputum])
    else:
        symptom_questionnaire = ",".join(
            [symptom_questionnaire, patient_symptom_MRC_dyspnea_scale, patient_symptom_sputum])

    if patient_symptom_sputum == "no":
        symptom_questionnaire = ",".join([symptom_questionnaire, ""])
    else:
        symptom_questionnaire = ",".join([symptom_questionnaire, patient_symptom_sputum_color])

    # the rest of the symptoms
    symptom_questionnaire = ",".join([symptom_questionnaire,
                                      patient_vesicular_breathing_sound,
                                      patient_bronchial_breathing_sound,
                                      patient_decreased_breathing_sound,
                                      patient_localization_of_decreased_breathing_sound,
                                      patient_coarse_bubble_rattling_sound,
                                      patient_localization_of_coarse_bubble_rattling_sound,
                                      patient_fine_bubble_rattling_sound,
                                      patient_localization_of_fine_bubble_rattling_sound,
                                      patient_crackling_rattle_sound,
                                      patient_localization_of_crackling_rattle_sound,
                                      patient_hum,
                                      patient_whistling_wheezing_inspiratory,
                                      patient_whistling_wheezing_expiratory,
                                      patient_stridor])

    final_csv = csv_header + "\n" + ",".join([demographic_data, clinical_data_and_medical_history,
                                              symptom_questionnaire])
    b64 = base64.b64encode(final_csv.encode()).decode()
    href = f'<a href="data:file/patient_{patient_pseudonym};base64,{b64}" download="patient_{patient_pseudonym}.csv"' \
           f'>** ⯈ Download patient_{patient_pseudonym}.csv**</a>'
    st.markdown(href, unsafe_allow_html=True)


def calculate_bmi(patient_height, patient_weight):
    if float(patient_height) == 0.0 and float(patient_weight) == 0.0:
        return np.NaN
    else:
        return float(patient_weight) / ((float(patient_height) / 100.0) * (float(patient_height) / 100.0))
