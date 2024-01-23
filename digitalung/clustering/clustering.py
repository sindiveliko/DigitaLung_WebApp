"""Time Series Clustering

This module provides all functions for clustering pandas Dataframes.

This module contains the following functions:
    * kmeans_clustering - clustering using tslearn TimeSeriesKMeans
    * kernel_means_clustering - clustering using tslearn KernelKMeans
    * kShape_clustering - clustering using tslearn kShape
"""

import os
from collections import Counter
from copy import deepcopy
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score
from tslearn.clustering import TimeSeriesKMeans, silhouette_score, KernelKMeans, KShape
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax

from general_utilities.utilities import from_sktime_df_to_tslearn_ndarray, from_csv_df_list_to_sktime_df, \
    str_percentage_of_elements_in_clusters

from config import seed


@st.cache_resource(show_spinner=False)
@st.cache_data(show_spinner=False)
def prep_clust_data(df_list, dimensionality_type, n_seg, normalization_type):
    """
    Apply normalization techniques (Z-Score, Min-Max, None) and/or PAA.
    Prep the files in tslearn format for clustering.
    :param df_list: List of time series in dataframe format
    :param dimensionality_type: 'None' or 'PAA'
    :param n_seg: Number of segments to use if PAA is chosen.
    :param normalization_type: 'None', 'Z-Score' or 'Min-Max'
    :return: Dataset prepped in (n_ts, sz, d) format for clustering
    """
    s = timer()
    progress_bar = st.progress(0, text='Prepping files')
    # Remove timestamp column from the dataframe since they are not needed for clustering
    df_ls = deepcopy(df_list)
    df_l = []  # #* len(df_list) = 920
    for df in df_ls:
        del df[df.columns[0]]
        df_l.append(df)

    # Transform a sktime-compatible dataset into a tslearn dataset, after creating a sktime compatible dataset
    # from the df list
    progress_bar.progress(25, text='Dataset Format')
    tslearn_nda_train = from_sktime_df_to_tslearn_ndarray(from_csv_df_list_to_sktime_df(df_l))

    # Normalization
    if normalization_type == 'Z-Score':
        progress_bar.progress(50, text='Z-Score Normalization')
        print(' --> Z-Score Normalization')
        scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
        tslearn_nda_train = scaler.fit_transform(tslearn_nda_train)
    elif normalization_type == 'Min-Max':
        progress_bar.progress(50, text='Min-Max Normalization')
        print(' --> Min-Max Normalization')
        scaler = TimeSeriesScalerMinMax(value_range=(-1., 1.))  # Rescale time series
        tslearn_nda_train = scaler.fit_transform(tslearn_nda_train)
    else:
        progress_bar.progress(50, text='No Normalization')
        print(' --> No Normalization')

    # Dimensionality Reduction
    progress_bar.progress(75, text='Aggregate Approximation')
    if dimensionality_type == 'None':
        dataset = tslearn_nda_train
        print(' --> No PAA')
        progress_bar.progress(100, text=f'Dataset has shape (nr_files x length x d): {dataset.shape}. '
                                        f'Methods: Normalization Type = {normalization_type}, '
                                        f'PAA = {dimensionality_type}')
    else:
        paa_object = PiecewiseAggregateApproximation(n_segments=n_seg)
        dataset = paa_object.fit_transform(tslearn_nda_train)  # Dimensionality Reduction
        print(' --> PAA segments: ', n_seg)
        progress_bar.progress(100, text=f'Dataset has shape (nr_files x length x d): {dataset.shape}. '
                                        f'Methods: Normalization Type = {normalization_type}, '
                                        f'PAA = {dimensionality_type} ({n_seg} segments)')

    print('Time for dataset preparation in seconds ----', timer() - s)
    return dataset


@st.cache_data(show_spinner=False)
@st.cache_resource(show_spinner=False)
def kmeans_clustering(tslearn_nda_train, distance_function, sakoe_state, sakoe, num_clusters,
                      initial_centroid_input, labels_list, cycles_file_names, cycles_nr):
    """
    Clusters a list of pandas Dataframes using tslearn k-Means.
    :param tslearn_nda_train: Dataset in format (n_ts, sz, d)
    :param distance_function: A String of the distance function that should be used for the clustering
    :param sakoe_state: True if Sakoe-Chiba constrain is applied, False if no constrain is applied
    :param sakoe: The radius (in steps) to use for Sakoe-Chiba parameter
    :param num_clusters: Number of k
    :param initial_centroid_input: Method used to initialize the model
    :param labels_list: List of labels corresponding to ts
    :param cycles_nr: Cycle positions within the recording
    :param cycles_file_names: File name corresponding to cycle
    """
    print('--------------------------------------------------------')
    print('--> Dataset has shape: ', tslearn_nda_train.shape)

    output_placeholder = st.empty()
    output_messages = [f"Clustering dataset with shape: {tslearn_nda_train.shape}"]
    output_placeholder.write(output_messages[0])

    # Utilize all CPU
    n_cpu = os.cpu_count()
    n_jobs = n_cpu
    print('--> ', n_cpu, 'is the number of cpus')
    output_messages.append(f"\nNr. of CPUs (nr. of parallel jobs) = {n_cpu}")
    output_placeholder.write("\n".join(output_messages))

    # Initialization Methods        # ## K-Means ++
    output_messages.append(f"\nInitialization method = {initial_centroid_input}")
    output_placeholder.write("\n".join(output_messages))
    init = 'k-means++'
    if distance_function == 'Euclidean':
        n_init = 3
    else:  # DTW
        n_init = 1

    # K-Means Model
    output_messages.append(f"\nDistance function = {distance_function}")
    output_placeholder.write("\n".join(output_messages))
    print(distance_function, "k-Means")

    if distance_function == "Euclidean":
        max_iter = 300
        km = TimeSeriesKMeans(n_clusters=num_clusters,
                              metric="euclidean",
                              n_jobs=n_jobs,
                              n_init=n_init,
                              max_iter=max_iter,
                              verbose=True,
                              random_state=seed,
                              init=init)

        output_messages.append(f"\nnr_init = {n_init} ; max_iter = {max_iter} ; k = {num_clusters}")
        output_placeholder.write("\n".join(output_messages))
        print("n_init = ", n_init, " ; max_iter = ", max_iter, " ; k = ", num_clusters)

    else:  # distance_function == "DTW"
        max_iter = 10
        max_iter_barycenter = 10

        if sakoe_state is True:
            sakoe_radius_steps = np.int32((sakoe / 100) * tslearn_nda_train.shape[1])
            km = TimeSeriesKMeans(n_clusters=num_clusters,
                                  metric="dtw",
                                  n_jobs=n_jobs,
                                  n_init=n_init,
                                  max_iter=max_iter,
                                  max_iter_barycenter=max_iter_barycenter,  # 5,
                                  metric_params={'global_constraint': 'sakoe_chiba',
                                                 'sakoe_chiba_radius': sakoe_radius_steps},
                                  # Add Sakoe_Chiba
                                  verbose=True,
                                  random_state=seed,
                                  init=init)
            output_messages.append(f'\nNot Zero, Constrain added =>  '
                                   f'radius of sakoe_chiba = {sakoe}% = {sakoe_radius_steps} steps')
            output_placeholder.write("\n".join(output_messages))
            print(f'Not Zero, Constrain added =>  r = {sakoe}% = {sakoe_radius_steps} steps')

        else:  # No constrain
            km = TimeSeriesKMeans(n_clusters=num_clusters,
                                  metric="dtw",
                                  n_jobs=n_jobs,
                                  n_init=n_init,
                                  max_iter=max_iter,
                                  max_iter_barycenter=max_iter_barycenter,
                                  verbose=True,
                                  random_state=seed,
                                  init=init)

            output_messages.append('\nZero, No Constrain (radius of sakoe_chiba)')
            output_placeholder.write("\n".join(output_messages))
            print('Zero, No Constrain')

        output_messages.append(f"\nnr_init = {n_init} ; max_iter = {max_iter} ; "
                               f"max_iter_barycenter = {max_iter_barycenter} ; k = {num_clusters}")
        output_placeholder.write("\n".join(output_messages))
        print("n_init / max_iter / max_iter_barycenter / k => ", n_init, max_iter, max_iter_barycenter, num_clusters)

    # Fit and Predict
    start = timer()
    with st.spinner("Clustering ... "):
        y_pred = km.fit_predict(tslearn_nda_train)
    print('Time for ypred ----', timer() - start)
    output_messages.append(f'\nPrediction finished in {round((timer() - start) / 60, 3)} minutes.')
    output_placeholder.write("\n".join(output_messages))
    st.markdown('---')

    # OUTPUT: CLUSTER CARDINALITY
    # Generate a list of unique values from 0 to num_clusters-1 = possible cluster labels
    values = _km_cardinality(km, num_clusters, tslearn_nda_train, y_pred, labels_list)

    # OUTPUT: METADATA
    _km_metadata(num_clusters, tslearn_nda_train, y_pred, cycles_file_names, cycles_nr)

    # OUTPUT: CENTROIDS
    _wav_visualize_clusters_with_centroid(km, num_clusters, tslearn_nda_train, y_pred, cycles_file_names,
                                          labels_list, cycles_nr)

    # Evaluation Metrics - Borrowed from Supervised
    _evaluate_km_external(km, num_clusters, labels_list, values)

    # Evaluation Metrics - Internal
    st.info(f'**Iterations** run: {km.n_iter_}')
    st.info(f'**Inertia** is {km.inertia_:.2f}')
    print('Iterations run / Inertia: ', km.n_iter_, km.inertia_)
    start = timer()
    with st.spinner('Computing Silhouette...'):
        # Compute the mean Silhouette Coefficient of all samples
        score = silhouette_score(tslearn_nda_train, y_pred, metric=distance_function.lower(), n_jobs=n_jobs)
    st.info(f'**Silhouette Score** is {score:.2f}')
    print('Time for Silhouette_Score ----', timer() - start)
    print('Silhouette Score : ', score)


@st.cache_data(show_spinner=False)
@st.cache_resource(show_spinner=False)
def kernel_means_clustering(tslearn_nda_train, kernel_function, num_clusters,
                            labels_list, cycles_file_names, cycles_nr):
    """Clusters a list of pandas Dataframes using tslearn KernelKMeans.

        :param tslearn_nda_train: Dataset in format (n_ts, sz, d)
        :param kernel_function: A String of the distance function that should be used for the clustering
        :param num_clusters:  k
        :param labels_list: Labels corresponding to ts
        :param cycles_file_names: File name corresponding to cycle
        :param cycles_nr: Cycle positions within the recording
        """
    output_placeholder = st.empty()
    output_messages = [f"KernelKMeans with Kernel: {kernel_function}"]
    output_placeholder.write(output_messages[0])

    n_cpu = os.cpu_count()
    n_jobs = n_cpu
    print('--> ', n_cpu, 'is the number of cpus')
    output_messages.append(f"\nNr. of CPUs (nr. of parallel jobs) = {n_cpu}")
    output_placeholder.write("\n".join(output_messages))

    if kernel_function == 'gak':
        km = KernelKMeans(n_clusters=num_clusters,
                          kernel=kernel_function,
                          n_jobs=n_jobs,
                          kernel_params={"sigma": "auto"},
                          n_init=2,
                          max_iter=5,
                          verbose=True,
                          random_state=seed)
    else:
        km = KernelKMeans(n_clusters=num_clusters,
                          kernel=kernel_function,
                          n_jobs=n_jobs,
                          n_init=2,
                          max_iter=5,
                          verbose=True,
                          random_state=seed)

    start = timer()
    with st.spinner('Clustering ... '):
        y_pred = km.fit_predict(tslearn_nda_train)
    print('Time for prediction: ', timer() - start)
    output_messages.append(f'\nPrediction finished in {round((timer() - start) / 60, 3)} minutes.')
    output_placeholder.write("\n".join(output_messages))
    st.markdown('---')

    # OUTPUT: CLUSTER CARDINALITY
    # Generate a list of unique values from 0 to num_clusters-1 = possible cluster labels
    values = _km_cardinality(km, num_clusters, tslearn_nda_train, y_pred, labels_list)

    # OUTPUT: METADATA
    _km_metadata(num_clusters, tslearn_nda_train, y_pred, cycles_file_names, cycles_nr)

    # OUTPUT: CENTROIDS
    _visualize_clusters_without_centroid(num_clusters, tslearn_nda_train, y_pred, cycles_file_names,
                                         labels_list, cycles_nr)

    # Evaluation Metrics - Borrowed from Supervised
    _evaluate_km_external(km, num_clusters, labels_list, values)

    # Evaluation Metrics - Internal
    st.info(f'**Iterations** run: {km.n_iter_}')
    st.info(f'**Inertia** is {km.inertia_:.2f}')
    print('Iterations run / Inertia / SS : ', km.n_iter_, km.inertia_)


@st.cache_data(show_spinner=False)
@st.cache_resource(show_spinner=False)
def kshape_clustering(tslearn_nda_train, num_clusters, labels_list, cycles_file_names, cycles_nr):
    """Clusters a list of pandas Dataframes using tslearn kShape.

        :param labels_list: List of labels corresponding to ts
        :param tslearn_nda_train: tslearn_nda_train: Dataset in format (n_ts, sz, d)
        :param num_clusters: Number of Clusters that should be created from the clustering algorithm
        :param cycles_nr: Cycle positions within the recording
        :param cycles_file_names: File name corresponding to cycle
        """

    st.markdown('Start kShape.')
    km = KShape(n_clusters=num_clusters, verbose=True, random_state=seed)

    start = timer()
    with st.spinner('Clustering ... '):
        y_pred = km.fit_predict(tslearn_nda_train)
    st.markdown(f'Prediction finished in {round((timer() - start) / 60, 3)} minutes.')

    # OUTPUT: CLUSTER CARDINALITY
    # Generate a list of unique values from 0 to num_clusters-1 = possible cluster labels
    values = _km_cardinality(km, num_clusters, tslearn_nda_train, y_pred, labels_list)

    # OUTPUT: METADATA
    _km_metadata(num_clusters, tslearn_nda_train, y_pred, cycles_file_names, cycles_nr)

    # OUTPUT: CENTROIDS
    _wav_visualize_clusters_with_centroid(km, num_clusters, tslearn_nda_train, y_pred, cycles_file_names,
                                          labels_list, cycles_nr)

    # Evaluation Metrics - Borrowed from Supervised
    _evaluate_km_external(km, num_clusters, labels_list, values)

    # Evaluation Metrics - Internal
    st.info(f'**Iterations** run: {km.n_iter_}')
    st.info(f'**Inertia** is {km.inertia_:.2f}')
    print('Iterations run / Inertia / SS : ', km.n_iter_, km.inertia_)


def _km_cardinality(km, num_clusters, tslearn_nda_train, y_pred, labels_list):
    L = [[0] for _ in range(num_clusters)]  # #* L = [[0], [0]]
    # set the initial count of instances belonging to that cluster to zero

    # Iterate over cluster labels and increment the count of instances in the corresponding cluster i
    for i in km.labels_:
        L[i][0] = L[i][0] + 1

    st.info(f" - Number of instances: {len(km.labels_)}  \n"
            f" - Instances belonging to individual clusters after clustering: {L}  \n"  # #* [[12], [36]]
            f" - Percentage of instances in each cluster: {str_percentage_of_elements_in_clusters(len(km.labels_), L)}")

    print("Number of instances: ", len(km.labels_))
    print("Instances belonging to individual clusters after clustering: ", L)  # #* [[12], [36]]
    print("Percentage of instances in each cluster: ", str_percentage_of_elements_in_clusters(len(km.labels_), L))

    cluster_c = [item for sublist in L for item in sublist]  # c: count
    percent_c = [value / sum(cluster_c) * 100 for value in cluster_c]
    cluster_n = ["Cluster " + str(i + 1) for i in range(num_clusters)]  # n:names

    f, ax = plt.subplots(figsize=(8, 2))
    ax.set_title("Cluster Cardinality", fontsize=10)
    bars = ax.bar(cluster_n, percent_c)
    # Adding the percentage labels on top of each bar
    for bar, percentage in zip(bars, percent_c):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f'{percentage:.1f}%',
                 ha='center',
                 color='black',
                 fontsize=8)
    ax.set_ylim(0, max(percent_c) + 20)
    ax.tick_params(axis='both', which='major', labelsize=8)
    st.pyplot(f, use_container_width=False)

    # OUTPUT: LABEL DISTRIBUTION WITHIN EACH CLUSTER
    with st.spinner("Counting respiratory sounds based on known label ..."):
        values = []
        for yi in range(num_clusters):
            label_counter = Counter()
            for index, xx in enumerate(tslearn_nda_train):
                if y_pred[index] == yi:
                    cycle_label = labels_list[index]
                    label_counter[cycle_label] += 1
            values.extend([label_counter['N'], label_counter['C'], label_counter['W'], label_counter['B']])
        print(" N/C/W/B per cluster or N/AB per cluster: ")
        print(values)  # [[n,c,w,b],[n,c,w,b]]

    # Y-Axis: Number of files per cluster
    fig, axs = plt.subplots(1, num_clusters, figsize=(16, 3))
    for yi in range(num_clusters):
        categories = ['N', 'C', 'W', 'B']
        n_catg = len(categories)
        rc_cycle = values[yi * n_catg:yi * n_catg + n_catg]
        bar_container = axs[yi].bar(categories, rc_cycle, color=['tab:green', 'tab:olive', 'tab:orange', 'tab:red'])
        axs[yi].set_title(f'Cluster {yi + 1}', fontsize=10)
        axs[yi].set_ylim(ymax=max(values) + 20)
        axs[yi].set_ylabel('Files')
        axs[yi].bar_label(bar_container, fmt='{:,.0f}')
    st.pyplot(fig, use_container_width=False)

    # Y-Axis: Percentage of label presence within the respective cluster
    fig, axs = plt.subplots(1, num_clusters, figsize=(16, 3))
    for yi in range(num_clusters):
        categories = ['N', 'C', 'W', 'B']
        n_catg = len(categories)
        rc_cycle = values[yi * n_catg:yi * n_catg + n_catg]
        total_values_cluster = sum(rc_cycle)
        rc_cycle = [(value / total_values_cluster) * 100 for value in rc_cycle]
        print('Cluster', yi + 1, 'has distribution in %:', rc_cycle)
        bar_container = axs[yi].bar(categories, rc_cycle, color=['tab:green', 'tab:olive', 'tab:orange', 'tab:red'])
        axs[yi].set_title(f'Cluster {yi + 1}', fontsize=10)
        axs[yi].set_ylim(ymax=max(rc_cycle) + 20)
        axs[yi].set_ylabel('Percentage')
        axs[yi].bar_label(bar_container, fmt='{:,.0f}%')
    st.pyplot(fig, use_container_width=False)

    return values


def _km_metadata(num_clusters, tslearn_nda_train, y_pred, cycles_file_names, cycles_nr):
    cycles_df_metadata = pd.read_csv('./ICBHI_dataset/cycles_df_metadata.csv')

    identified_file_names = []
    for yi in range(num_clusters):
        cluster_files = []
        for xx in tslearn_nda_train[y_pred == yi]:
            index = (np.where(np.all(tslearn_nda_train == xx, axis=1))[0])[0]
            cluster_files.append([cycles_file_names[index], cycles_nr[index]])
        identified_file_names.append(cluster_files)

    clusters_df_metadata = []
    for i in range(num_clusters):
        clust_files = identified_file_names[i]
        new_df = pd.DataFrame()
        for file in clust_files:
            filtered_rows = cycles_df_metadata[(cycles_df_metadata['File Name'] == file[0]) &
                                               (cycles_df_metadata['Cycle Position'] == file[1])]
            new_df = pd.concat([new_df, filtered_rows], ignore_index=False)
        clusters_df_metadata.append(new_df)

    # Display dataframe
    column_names = st.columns(num_clusters)
    for i, name in enumerate(column_names):
        with name:
            st.markdown(f'##### Cluster Nr. {i + 1}')
            st.dataframe(clusters_df_metadata[i])

    # Display numerical statistics
    column_names = st.columns(num_clusters)
    for i, name in enumerate(column_names):
        with name:
            numerical_statistics = {}
            for col in clusters_df_metadata[i].columns:
                if clusters_df_metadata[i][col].dtype in ['int64', 'float64', 'int32', 'float32'] and \
                        not clusters_df_metadata[i][col].isna().all():
                    nan_count = clusters_df_metadata[i][col].isna().sum()

                    if nan_count > 0:
                        numerical_statistics[col] = {
                            'NaN Count': nan_count,
                            'Mean': clusters_df_metadata[i][col].mean(),
                            'Median': clusters_df_metadata[i][col].median(),
                            'Standard Deviation': clusters_df_metadata[i][col].std()
                        }
                    else:
                        numerical_statistics[col] = {
                            'Mean': clusters_df_metadata[i][col].mean(),
                            'Median': clusters_df_metadata[i][col].median(),
                            'Standard Deviation': clusters_df_metadata[i][col].std()
                        }

            st.subheader("Numerical Statistics")
            num_stats_df = pd.DataFrame(numerical_statistics)
            st.write(num_stats_df)

    # Display categorical statistics
    column_names = st.columns(num_clusters)
    for i, name in enumerate(column_names):
        with name:
            categorical_statistics = {}
            for col in clusters_df_metadata[i].columns:
                if clusters_df_metadata[i][col].dtype == 'object' and not clusters_df_metadata[i][col].isna().all():
                    nan_count = clusters_df_metadata[i][col].isna().sum()

                    if nan_count > 0:
                        value_counts = clusters_df_metadata[i][col].value_counts(normalize=True) * 100  # in percentages
                        value_counts = value_counts.round(1)
                        categorical_statistics[col] = {
                            'NaN Count': nan_count,
                            'Value Counts (%)': value_counts
                        }
                    else:
                        value_counts = clusters_df_metadata[i][col].value_counts(normalize=True) * 100  # in percentages
                        value_counts = value_counts.round(1)
                        categorical_statistics[col] = {
                            'Value Counts (%)': value_counts
                        }

            st.markdown("###### Categorical Statistics")
            cat_stats_df = pd.DataFrame(categorical_statistics)
            st.write(cat_stats_df.astype(str))


def _wav_visualize_clusters_with_centroid(km, num_clusters, tslearn_nda_train, y_pred, cycles_file_names,
                                          cycles_labels_list, cycles_nr):
    for yi in range(num_clusters):
        fig = go.Figure()
        instance_per_cluster = 0
        for xx in tslearn_nda_train[y_pred == yi]:
            index = (np.where(np.all(tslearn_nda_train == xx, axis=1))[0])[0]
            file_name = cycles_file_names[index]
            label = cycles_labels_list[index]
            cycle_nr = cycles_nr[index]
            name = f'{cycle_nr} - {label} - {file_name}'
            # break loop if more than 20 instances
            if instance_per_cluster > 20:
                st.write(
                    f"Too many instances in cluster {yi + 1}. Only {instance_per_cluster} instances will be displayed, "
                    f"but all instances were considered for the centroid calculation.")
                break
            flattened = xx.ravel()  #
            indices = list(range(len(flattened)))
            fig.add_trace(go.Scatter(
                x=indices, y=flattened, mode='lines', visible='legendonly',
                opacity=0.2,
                line=dict(
                    color='black',
                    width=2
                ),
                name=name,
            ))
            instance_per_cluster += 1
        centroid = km.cluster_centers_[yi].ravel()
        fig.add_trace(go.Scatter(
            x=indices, y=centroid, mode='lines', name="centroid", line=dict(
                color='red',
                width=2
            ),
            showlegend=True
        ))

        fig.update_layout(height=400, width=1500, title_text="Cluster: " + str(yi + 1))
        st.plotly_chart(fig, use_container_width=False)


def _visualize_clusters_without_centroid(num_clusters, tslearn_nda_train, y_pred, cycles_file_names,
                                         cycles_labels_list, cycles_nr):
    for yi in range(num_clusters):
        fig = go.Figure()
        for cluster_entry in tslearn_nda_train[y_pred == yi]:
            ts_index = (np.where(np.all(tslearn_nda_train == cluster_entry, axis=1))[0])[0]
            file_name = cycles_file_names[ts_index]
            label = cycles_labels_list[ts_index]
            cycle_nr = cycles_nr[ts_index]
            name = f'{cycle_nr} - {label} - {file_name}'
            flattened = cluster_entry.ravel()
            indices = list(range(len(flattened)))
            fig.add_trace(go.Scatter(
                x=indices, y=flattened, mode='lines',
                opacity=0.2,
                line=dict(
                    color='black',
                    width=2
                ),
                name=name,
                showlegend=True
            ))
        fig.update_layout(height=600, width=1300, title_text="Cluster: " + str(yi + 1))
        st.plotly_chart(fig, use_container_width=True)


def _evaluate_km_external(km, num_clusters, labels_list, values):
    st.markdown('---')
    predicted_labels = km.labels_  # predicted cluster labels
    # The mapping of position to label.
    if num_clusters == 4:
        label_mapping = {'N': 0, 'C': 1, 'W': 2, 'B': 3}  # n c w b
        labels = [label_mapping[label_str] for label_str in labels_list]

        cluster_label = ['C1', 'C2', 'C3', 'C4']
        row_label = ['N', 'C', 'W', 'B']
        table_matrix = np.zeros((4, 4), dtype=int)
        for i in range(len(labels)):
            table_matrix[labels[i]][predicted_labels[i]] += 1

        st.write(" ----- Files / Cluster -----")
        table_matrix = pd.DataFrame(table_matrix, index=row_label, columns=cluster_label)
        st.table(table_matrix)

        # Adjusted Rand Index
        ari = adjusted_rand_score(labels, predicted_labels)
        st.info(f"**Adjusted Rand Index** is {ari:.6f}")

        # Cluster purity
        total_purity = 0
        purity_output = ''
        for i in range(num_clusters):
            cluster_data = values[i * 4: (i + 1) * 4]
            cl_purity = max(cluster_data) / sum(cluster_data)
            purity_output += f'**Cluster {i + 1}** has **Purity** {cl_purity:.2f}  \n'
            total_purity = total_purity + cl_purity
        avg_purity = total_purity / num_clusters
        purity_output += f'**Purity Average** is {avg_purity:.2f}'
        st.info(purity_output)

    elif num_clusters == 2:
        labels = [1 if x != 'N' else 0 for x in labels_list]  # regroup for new ground truth: Normal and Abnormal
        cluster_label = ['C1', 'C2']
        row_label = ['Normal', 'Adventitious']

        # Compute the table matrix
        table_matrix = np.zeros((2, 2), dtype=int)
        for i in range(len(labels)):
            table_matrix[labels[i]][predicted_labels[i]] += 1

        st.write(" ----- Files / Cluster -----")
        table_matrix = pd.DataFrame(table_matrix, index=row_label, columns=cluster_label)
        st.table(table_matrix)

        # Adjusted Rand Index
        ari = adjusted_rand_score(labels, predicted_labels)
        st.info(f"**Adjusted Rand Index** is {ari:.6f}")

        # Cluster purity
        cluster_data1 = values[0: (0 + 1) * 4]
        cluster_data_1 = [cluster_data1[0], sum(cluster_data1[1:4])]
        cl_purity1 = max(cluster_data_1) / sum(cluster_data_1)

        cluster_data2 = values[4: 8]
        cluster_data_2 = [cluster_data2[0], sum(cluster_data2[1:4])]
        cl_purity2 = max(cluster_data_2) / sum(cluster_data_2)

        total_purity = cl_purity1 + cl_purity2
        avg_purity = total_purity / 2

        st.info(f'**Cluster 1** has **Purity** {cl_purity1:.2f}  \n'
                f'**Cluster 2** has **Purity** {cl_purity2:.2f}  \n'
                f'**Purity Average** is {avg_purity:.2f}')
