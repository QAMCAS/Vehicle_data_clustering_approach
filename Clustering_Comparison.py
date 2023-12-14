import math
import time
import pandas as pd
import weka.core.jvm as jvm
from keras import Sequential
from keras.layers import Input, Dense, LSTM
from keras.models import Model
import traceback
from weka.filters import Filter
from sklearn.cluster import DBSCAN
from weka.clusterers import ClusterEvaluation, Clusterer
from weka.core.converters import Loader
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from Qualitative_abstraction import Compute_PCORR_score
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
from sklearn.metrics import silhouette_score
from dtaidistance import dtw, clustering

# logging setup
logger = logging.getLogger(__name__)


def main():
    # reading data
    loader = Loader("weka.core.converters.ArffLoader")
    datasets = ["Gaimersheim", "Ingolstadt", "Munich"]

    for location in datasets:
        path_to_clean_data = os.path.join(
            "data/Clustering_input_data/", location, "test_data_cleaned.arff")
        path_to_filtered_data = os.path.join(
            "data/Clustering_input_data/", location, "test_data_Filtered.arff")

        clean_data = loader.load_file(path_to_clean_data)
        clean_df = pd.DataFrame(clean_data, columns=clean_data.attribute_names())
        input_data = loader.load_file(path_to_filtered_data)
        X = pd.DataFrame(input_data, columns=input_data.attribute_names())

        nb_clusters = [4, 8]  # it is possible to add more numbers of clusters
        for k in nb_clusters:
            results = {'Method': [],
                       'avg PCORR': [],
                       'avg Silhouette_Score': [],
                       'DB_index': [],
                       'CH_index': [],
                       'Clustering_time(s)': [],
                       }
            # runningg Time series k-means
            TS_Kmeans_labels, TS_Kmeans_clustering_time = TSkmeans_labels(X, k)

            # Need to reshape the data for both LSTN and the LSTM Auto-encoder
            # Assuming X is either a 2D array (num_samples, num_features) or a 3D array (num_samples, num_timesteps,
            # num_features)
            print(len(X.shape))
            if len(X.shape) == 2:
                num_samples, num_features = X.shape
                num_timesteps = 1  # Assuming a single time step for 2D data
            else:
                num_samples, num_timesteps, num_features = X.shape

            # Reshape the input data to be 3D if it's 2D
            if len(X.shape) == 2:
                X_reshaped = X.values.reshape((num_samples, 1, num_features))

            # running LSTM combined with k-means
            KM_LSTM_labels, KM_LSTM_clustering_time = LSTM_Clustering(X_reshaped, k)
            # running LSTM auto-encoder combined with k-means
            KM_LSTM_autoencoder_labels, KM_LSTM_autoencoder_clustering_time = LSTM_Autoencoder(X_reshaped,
                                                                                               num_timesteps,
                                                                                               num_features,
                                                                                               k)
            # running k-means with Dynamic Time Wrapping measure
            KM_DTW_labels, KM_DTW_clustering_time = DTW_KM(X, input_data, k)

            # running DBBSCAN clustering
            dbscan_labels, dbscan_clustering_time = DBSCAN_clustering(X)

            # Evaluation of the different apporoach usingg Silhouette score, DB index and CH index

            weka_clusterers = ['SimpleKMeans', 'EM', 'Canopy', 'Kmeans with Autoencoder',
                               'SelfOrganizingMap']  # these are the algorithms provided by WEKA library
            Additional_methods = ["TS Kmeans", "kmeans with LSTM", "Kmeans with LSTM Auto-encoder", "Kmeans with DTW",
                           "DBSCAN"]
            new_methods_Labels = [TS_Kmeans_labels, KM_LSTM_labels, KM_LSTM_autoencoder_labels, KM_DTW_labels,
                                  dbscan_labels]
            new_methods_clustering_time = [TS_Kmeans_clustering_time, KM_LSTM_clustering_time,
                                           KM_LSTM_autoencoder_clustering_time,
                                           KM_DTW_clustering_time, dbscan_clustering_time]

            df_pcorr = pd.DataFrame(input_data, columns=input_data.attribute_names())

            if k != 4: weka_clusterers.remove(
                'SelfOrganizingMap')  # because 'SelfOrganizingMap' can only generates 4 clusters

            for method in weka_clusterers:
                results['Method'].append(method)

                print(method)
                labels, clustering_time, df_metrics = WEKA_Clustering(input_data, k, method)

                PCORR, silhouette_avg, DB_index, CH_index = clustering_evaluation(clean_df, df_metrics, df_pcorr,
                                                                                  method, labels,
                                                                                  k)
                results['avg PCORR'].append(PCORR)
                results['avg Silhouette_Score'].append(round(silhouette_avg, 4))
                results['DB_index'].append(round(DB_index, 4))
                results['CH_index'].append(int(CH_index) / 1000)
                results['Clustering_time(s)'].append(round(clustering_time, 4))

            for method, new_labels, Cl_time in zip(Additional_methods, new_methods_Labels, new_methods_clustering_time):
                print(method)
                df_metrics = pd.DataFrame(input_data, columns=input_data.attribute_names())

                PCORR, silhouette_avg, DB_index, CH_index = clustering_evaluation(clean_df, df_metrics, df_pcorr,
                                                                                  method,
                                                                                  new_labels, k)
                results['Method'].append(method)
                results['avg PCORR'].append(PCORR)
                results['avg Silhouette_Score'].append(round(silhouette_avg, 4))
                results['DB_index'].append(round(DB_index, 4))
                results['CH_index'].append(int(CH_index) / 1000)
                results['Clustering_time(s)'].append(round(Cl_time, 4))

            df_results = pd.DataFrame(results)
            print(
                df_results.to_latex(index=False,
                                    caption='Clustering comparison' + ' using ' + str(
                                        k) + ' clusters in ' + location))


# Visualize the clustered data for each technique
def plot_clusters(X, labels, title, nb_cluster):
    clusters_labels = [k for k in nb_cluster]
    plt.figure(figsize=(10, 6))
    plt.scatter(X['timestamps'], X['vehicle_speed'], c=labels, s=10, label=clusters_labels)
    plt.xlabel('Timestamps')
    plt.ylabel('Vehicle Speed')
    plt.legend()
    plt.title(title)
    plt.show()


# runs the clustering algorithms provided by WEKA
def WEKA_Clustering(input_data, nb_clusters, algorithm):
    start_time = time.time()

    if algorithm != 'SelfOrganizingMap' and algorithm != "Kmeans with Autoencoder":
        start_time = time.time()
        clusterer = Clusterer(classname="weka.clusterers." + str(algorithm), options=["-N", str(nb_clusters)])
        clusterer.build_clusterer(input_data)
        print(clusterer)
        evaluation = ClusterEvaluation()
        evaluation.set_model(clusterer)
        evaluation.test_model(input_data)
        print("eva;" + str(evaluation.cluster_results))
        print("# clusters: " + str(evaluation.num_clusters))
        print("log likelihood: " + str(evaluation.log_likelihood))
        print("cluster assignments:\n" + str(evaluation.cluster_assignments))
        df_metrics = pd.DataFrame(input_data, columns=input_data.attribute_names())


    elif algorithm == "Kmeans with Autoencoder":

        # clustering Using SimpkeKMeans with autoencoder preprocessing
        print("**** Clustering using Weka SimpkeKMeans with autoencoder ****")
        Autoencoder = Filter(classname="weka.filters.unsupervised.attribute.MLPAutoencoder")
        Autoencoder.inputformat(input_data)
        filtered = Autoencoder.filter(input_data)  # data filtered with  autoencoder
        df_metrics = pd.DataFrame(filtered, columns=filtered.attribute_names())

        clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", str(nb_clusters)])
        clusterer.build_clusterer(filtered)
        print(clusterer)
        evaluation = ClusterEvaluation()
        evaluation.set_model(clusterer)
        evaluation.test_model(filtered)
        print("evaluation;" + str(evaluation.cluster_results))
        print("# clusters: " + str(evaluation.num_clusters))
        print("log likelihood: " + str(evaluation.log_likelihood))
        print("cluster assignments:\n" + str(evaluation.cluster_assignments))


    elif algorithm == 'SelfOrganizingMap':
        start_time = time.time()
        clusterer = Clusterer(classname="weka.clusterers." + str(algorithm))  # for 'SelfOrganizingMap' you need
        # to remove cb_clusters]
        clusterer.build_clusterer(input_data)
        print(clusterer)
        evaluation = ClusterEvaluation()
        evaluation.set_model(clusterer)
        evaluation.test_model(input_data)
        print("evaluation;" + str(evaluation.cluster_results))
        print("# clusters: " + str(evaluation.num_clusters))
        print("log likelihood: " + str(evaluation.log_likelihood))
        print("cluster assignments:\n" + str(evaluation.cluster_assignments))
        df_metrics = pd.DataFrame(input_data, columns=input_data.attribute_names())

    end_time = time.time()  # Stop the timer
    clustering_time = end_time - start_time
    return evaluation.cluster_assignments, clustering_time, df_metrics


def LSTM_Clustering(X_reshaped, nb_clusters):
    start_time = time.time()
    # Define and train your LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])))

    # Extract features from LSTM layers
    features = model.predict(X_reshaped)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=nb_clusters, random_state=42)  # Specify the desired number of clusters
    LSTM_KM_labels = kmeans.fit_predict(features)
    end_time = time.time()  # Stop the timer
    clustering_time = end_time - start_time

    return LSTM_KM_labels, clustering_time


def LSTM_Autoencoder(X_reshaped, num_timesteps, num_features, nb_clusters):
    start_time = time.time()
    # Autoencoder for time series clustering
    input_shape = (num_timesteps, num_features)
    latent_dim = 20

    # Define the autoencoder model
    input_ts = Input(shape=input_shape)
    encoded = LSTM(latent_dim, activation='relu')(input_ts)
    decoded = Dense(np.prod(input_shape), activation='sigmoid')(encoded)
    autoencoder = Model(input_ts, decoded)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train the autoencoder
    autoencoder.fit(X_reshaped, X_reshaped, epochs=50, batch_size=64, validation_split=0.2)

    # Extract encoder part for clustering
    encoder = Model(input_ts, encoded)

    # Encode time series data
    X_encoded = encoder.predict(X_reshaped)

    # Apply K-Means on the encoded data
    kmeans_autoencoder = KMeans(n_clusters=nb_clusters, random_state=42)
    KM_LSTM_autoencoder_labels = kmeans_autoencoder.fit_predict(X_encoded)

    end_time = time.time()  # Stop the timer
    clustering_time = end_time - start_time

    return KM_LSTM_autoencoder_labels, clustering_time


def DTW_KM(X, input_data, nb_clusters):
    series_list = []
    for i in range(input_data.num_instances):
        instance = input_data.get_instance(i)
        # Assuming the time series attribute is the first one, adjust the index accordingly
        time_series = np.array(instance.values)
        series_list.append(time_series)
    start_time = time.time()
    KM_DTW = clustering.kmeans.KMeans(k=nb_clusters, max_it=10, max_dba_it=10, dists_options={"window": 40})
    KM_DTW_cluster_idx, performed_it = KM_DTW.fit(series_list, use_c=True, use_parallel=False)

    end_time = time.time()  # Stop the timer
    clustering_time = end_time - start_time

    # extracting labels for kmeans with DTW method
    KM_DTW_labels = [-1] * len(X)

    for cluster_id, indices in KM_DTW_cluster_idx.items():
        for instance_idx in indices:
            KM_DTW_labels[instance_idx] = cluster_id

    return KM_DTW_labels, clustering_time


def DBSCAN_clustering(X):
    # Specify the parameters for DBSCAN
    eps = 0.1  # Maximum distance between two samples for one to be considered as in the neighborhood of the other
    min_samples = 5  # The number of samples (or total weight) in a neighborhood for a point to be considered as a
    # core point

    start_time = time.time()
    # Create a DBSCAN object
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit the data and obtain labels
    dbscan_labels = dbscan.fit_predict(X)
    end_time = time.time()  # Stop the timer
    clustering_time = end_time - start_time

    return dbscan_labels, clustering_time


def TSkmeans_labels(X, nb_clusters):
    # TimeSeriesKMeans on time series data
    start_time = time.time()
    kmeans_time_series = TimeSeriesKMeans(n_clusters=nb_clusters, random_state=42)
    TSkmeans_labels = kmeans_time_series.fit_predict(X)
    end_time = time.time()  # Stop the timer
    clustering_time = end_time - start_time

    return TSkmeans_labels, clustering_time


def clustering_evaluation(clean_df, df_metrics, df_pcorr, method, labels, nb_cluster):
    plot_clusters(clean_df, labels, "Clustering using " + method, nb_cluster)
    DB_index = davies_bouldin_score(df_metrics, labels)
    CH_index = metrics.calinski_harabasz_score(df_metrics, labels)
    PCORR = Compute_PCORR_score(method, df_pcorr, labels, nb_cluster)
    silhouette_avg = silhouette_score(df_metrics, labels)
    print("Method", method, "For k =", nb_cluster, ": AVG silhouette_score  = ",
          round(silhouette_avg, 4), "PCORR = ",
          PCORR, "DB index = ", round(DB_index, 4),
          "and CH index =", int(CH_index))

    if math.isnan(PCORR):
        PCORR = 'nan'

    return PCORR, silhouette_avg, DB_index, CH_index


if __name__ == "__main__":
    try:
        jvm.start(system_cp=True, packages=True, max_heap_size="512m")
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
