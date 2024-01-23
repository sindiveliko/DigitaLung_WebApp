"""
Temporary for testing, not to be implemented in the final state.
"""
import json
import pandas as pd
import streamlit as st


def choose_subset_size_nr(subset_size, subset_nr, clust_data):
    """
    Extract subset based on chosen size and selection.
    Indexes of time series to be extracted are used to create a subset.
    :param subset_size: Size of subset in % (e.g., 5 for f% of full dataset)
    :param subset_nr: Subset within the desired range, choice "1" or "2"
    :param clust_data: full dataset to be selected from
    :return:
    """
    print(' --> ', subset_size, '% -- Nr', subset_nr)

    # Load subset dictionaries
    with open('./clustering/subsets/subsets_dict_chosen.json', 'r') as f:
        loaded_dict = json.load(f)
        index_list_size = loaded_dict[str(subset_size / 100)]
    # List of respiratory cycle time series' index in the database, corresponding to chosen subset
    index_list = index_list_size[subset_nr - 1]

    # Extract subset based on index list
    subset = [clust_data[i] for i in index_list]

    st.info(f'Size {subset_size}%  \n Subset Nr: {subset_nr}  \n'
            f'Nr. of cycles: {len(subset)}')

    # Extract cycles metadata
    df_cycles = pd.read_csv('./ICBHI_dataset/cycles_df_metadata.csv')
    subset_metadata = df_cycles.iloc[index_list]
    st.write(subset_metadata)

    # Extract required information
    labels_list = st.session_state.cycles_labels_list
    subset_labels_list = [labels_list[i] for i in index_list]

    cycle_position = st.session_state.cycles_nr
    subset_cycles_nr = [cycle_position[i] for i in index_list]

    file_names_cycles = st.session_state.cycles_file_names
    subset_cycles_file_names = [file_names_cycles[i] for i in index_list]

    return subset, subset_labels_list, subset_cycles_nr, subset_cycles_file_names
