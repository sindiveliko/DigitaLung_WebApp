"""Impute Pandas Dataframes

This module provides all functions necessary to impute Pandas Dataframes locally or globally.

This module contains the following functions:
    * sktime_impute - function that uses the sktime.transformations.series "Imputer" to impute data locally
    * own_imputation - our own imputations function that also does global imputations
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import streamlit as st
from sktime.transformations.series.impute import Imputer

from general_utilities import utilities


def local_imputation(dataframes, imputation_method):
    """Imputes Pandas Dataframes locally by using sktime.transformations.series Imputer.

    :param dataframes: A list containing all pandas.core.frame.DataFrame of the data
    :type dataframes: list
    :param imputation_method: A String representing the imputations method:
    :type imputation_method: str

    :returns: List: a list containing locally imputed pandas.core.frame.DataFrame timeseries in the sktime format
    """

    impute_seed = 0
    np.random.seed(impute_seed)

    df_list = _delete_object_columns_from_copied_dataframes(dataframes)

    sktime_df = utilities.from_wav_df_list_to_sktime_df(df_list)
    new_imputed_sktime_df = _impute_sktime_df(sktime_df, impute_method=imputation_method, random_state=impute_seed)
    imputed_sklearn_df_list = utilities.from_sktime_df_to_wav_df_list(new_imputed_sktime_df)

    st.markdown('---')
    count = 0
    for i, df in enumerate(imputed_sklearn_df_list):
        if df.isnull().values.any():
            count += 1
    if count > 0:
        st.warning(f"Imputation Method is local, there are still missing values. Try our own implementation methods, "
                   f"which are global.")

    return imputed_sklearn_df_list


def global_imputation(dataframes, imputation_method):
    """Imputes Pandas Dataframes globally.

    :param dataframes: A list containing all pandas.core.frame.DataFrame of the data
    :type dataframes: list
    :param imputation_method: A String representing the imputations method:
    :type imputation_method: str

    :returns: List: a list containing globally imputed pandas.core.frame.DataFrame timeseries in the sktime format
    """
    impute_seed = 0
    np.random.seed(impute_seed)

    df_list = _delete_object_columns_from_copied_dataframes(dataframes)

    # obtain sktime dfs from df_list_train and df_list_test
    sktime_df = utilities.from_wav_df_list_to_sktime_df(df_list)

    if imputation_method == "mean":
        new_imputed_sktime_df, means = _global_mean_impute_df(sktime_df)
    elif imputation_method == "median":
        new_imputed_sktime_df, medians = _global_median_impute_df(sktime_df)
    else:  # imputation_method == "zero":
        new_imputed_sktime_df = _global_zero_impute_df(sktime_df)

    imputed_sklearn_df_list = utilities.from_sktime_df_to_wav_df_list(new_imputed_sktime_df)

    count = 0
    for i, df in enumerate(imputed_sklearn_df_list):
        if df.isnull().values.any():
            count += 1
    if count > 0:
        st.warning("There are columns with only <NA> values. Please delete these columns before "
                   "trying the classification process.")
        st.stop()

    return imputed_sklearn_df_list


def _delete_object_columns_from_copied_dataframes(dataframes):
    # copy df_list before processing, otherwise dataframes will be changed, and that might produce errors
    df_list = deepcopy(dataframes)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Dimension and Type before Imputation:")
        for column in list(df_list[0]):
            st.write(f"{column}: {df_list[0][column].dtypes}")

    # deletes all columns with type object since they contain string values that cannot be imputed
    for i in range(len(df_list)):
        df_list[i] = df_list[i].select_dtypes(exclude='object')

    with col2:
        st.markdown("#### Dimension and Type after Imputation:")
        for column in list(df_list[0]):
            st.write(f"{column}: {df_list[0][column].dtypes}")

    return df_list


def _impute_sktime_df(df, impute_method, random_state=None):
    new_dict = {}

    for col in df.columns:
        df_dim = df[col]
        st.write("----------------")
        st.write("Dimension:", col)

        utilities.missing_value_count_in_nested_pd_series(df_dim, verbose=True)
        df_dim_imp = _impute_every_series_in_a_dim(df_dim, impute_method, random_state=random_state)
        utilities.missing_value_count_in_nested_pd_series(df_dim_imp, verbose=True)
        new_dict[col] = df_dim_imp

    new_df = pd.DataFrame(new_dict)
    return new_df


def _global_mean_impute_df(df, means=None, verbose=True):
    st.markdown("mean imputations started...")

    if means is None:
        means = {}

    for col in df.columns:
        df_dim = df[col]
        if verbose:
            st.markdown("----------------")
            st.write("Dimension:", col)

        if col not in means.keys():
            val_list = []
            for i in range(len(df_dim)):
                curr_series = list(df_dim[i])
                val_list.extend(curr_series)

            st.write("Len list with missing values:", len(val_list))
            val_list = np.asarray(val_list)
            val_list = val_list[~np.isnan(val_list)]
            st.write("Len list without missing values:", len(val_list))
            mean = np.mean(val_list)
            means[col] = mean

        utilities.missing_value_count_in_nested_pd_series(df_dim, verbose=verbose)
        # fill miss values
        for i in range(len(df_dim)):
            df_dim[i].fillna(value=means[col], inplace=True)
        utilities.missing_value_count_in_nested_pd_series(df_dim, verbose=verbose)

    st.markdown("mean imputations finished...")
    return df, means


def _global_median_impute_df(df, medians=None, verbose=True):
    st.markdown("median imputations started...")

    if medians is None:
        medians = {}

    for col in df.columns:
        df_dim = df[col]
        if verbose:
            st.markdown("----------------")
            st.write("Dimension:", f'{col}')

        if col not in medians.keys():
            val_list = []
            for i in range(len(df_dim)):
                curr_series = list(df_dim[i])
                val_list.extend(curr_series)

            st.write("Len list with missing values:", len(val_list))
            val_list = np.asarray(val_list)
            val_list = val_list[~np.isnan(val_list)]
            st.write("Len list without missing values:", len(val_list))
            median = np.median(val_list)
            medians[col] = median

        utilities.missing_value_count_in_nested_pd_series(df_dim, verbose=verbose)
        # fill miss values
        for i in range(len(df_dim)):
            df_dim[i].fillna(value=medians[col], inplace=True)
        utilities.missing_value_count_in_nested_pd_series(df_dim, verbose=verbose)

    st.markdown("median imputations finished...")
    return df, medians


def _global_zero_impute_df(df, verbose=True):
    st.markdown("zero imputations started...")

    for col in df.columns:
        df_dim = df[col]
        if verbose:
            st.markdown("----------------")
            st.write("Dimension:", f'{col}')

        utilities.missing_value_count_in_nested_pd_series(df_dim, verbose=verbose)
        # fill missing values
        for i in range(len(df_dim)):
            df_dim[i].fillna(value=0, inplace=True)
        utilities.missing_value_count_in_nested_pd_series(df_dim, verbose=verbose)

    st.markdown("zero imputations finished...")
    return df


def _impute_every_series_in_a_dim(df_dim, method, random_state=None):
    imp_list = []

    for i in range(len(df_dim)):
        imp = Imputer(method=method, random_state=random_state)
        imp.fit(df_dim[i])
        df_dim_ts = imp.transform(df_dim[i])
        imp_list.append(df_dim_ts)

    df_dim_imp = pd.Series(imp_list)
    return df_dim_imp
