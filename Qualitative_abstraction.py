import math
import os
import time
import numpy as np
import openpyxl
import pandas as pd
from matplotlib import pyplot as plt
import weka.core.jvm as jvm
import traceback
from weka.filters import Filter, MultiFilter
from weka.clusterers import Clusterer
from weka.clusterers import ClusterEvaluation
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
from sklearn.metrics import silhouette_score
from weka.core.converters import Loader
import PySimpleAutomata
import json
from PySimpleAutomata import DFA, automata_IO


def Speed_model1(speed_avg_list):
    Speed_Landmarks = []
    # using 2+- range
    Speed_Landmarks_ids = []
    for mean in speed_avg_list:
        if mean <= 2:
            Speed_Landmarks.append('0ff')
            Speed_Landmarks_ids.append(0)

        else:
            Speed_Landmarks.append('On')
            Speed_Landmarks_ids.append(1)

    return Speed_Landmarks, Speed_Landmarks_ids


def Speed_model2(speed_avg_list):
    Speed_Landmarks = []
    # using 2+- range
    Speed_Landmarks_ids = []
    for mean in speed_avg_list:
        if mean <= 2:
            Speed_Landmarks.append('S_0')
            Speed_Landmarks_ids.append(0)

        elif 2 < mean <= 50:
            Speed_Landmarks.append('S_0-S_50')
            Speed_Landmarks_ids.append(1)

        elif 50 < mean <= 80:
            Speed_Landmarks.append('S_50-S_80')
            Speed_Landmarks_ids.append(2)

        else:
            Speed_Landmarks.append('S_80')
            Speed_Landmarks_ids.append(3)

    return Speed_Landmarks, Speed_Landmarks_ids


def Speed_model3(speed_avg_list):
    Speed_Landmarks = []
    # using 2+- range
    Speed_Landmarks_ids = []
    for mean in speed_avg_list:
        if mean <= 2:
            Speed_Landmarks.append('S_0')
            Speed_Landmarks_ids.append(0)
        elif 2 < mean <= 20:
            Speed_Landmarks.append('S_0-S_20')
            Speed_Landmarks_ids.append(1)

        elif 20 < mean <= 40:
            Speed_Landmarks.append('S_20-S_40')
            Speed_Landmarks_ids.append(2)

        elif 40 < mean <= 60:
            Speed_Landmarks.append('S_40-S_60')
            Speed_Landmarks_ids.append(3)

        elif 60 < mean <= 80:
            Speed_Landmarks.append('S_60-S_80')
            Speed_Landmarks_ids.append(4)
        else:
            Speed_Landmarks.append('S_80')
            Speed_Landmarks_ids.append(5)

    return Speed_Landmarks, Speed_Landmarks_ids


def Steering_model1(Steering_avg_list):
    turning_threshold = 150
    Steering_Landmarks = []
    Steering_Landmarks_ids = []
    for s in Steering_avg_list:
        if s <= turning_threshold:
            Steering_Landmarks.append('Straight')
            Steering_Landmarks_ids.append(0)
        else:
            Steering_Landmarks.append('Turning')
            Steering_Landmarks_ids.append(1)

    return Steering_Landmarks, Steering_Landmarks_ids


def Steering_model2(Steering_avg_list_M2):
    turning_threshold = 150
    Steering_Landmarks = []
    Steering_Landmarks_ids = []

    for val in Steering_avg_list_M2:
        if -turning_threshold <= val <= turning_threshold:
            Steering_Landmarks.append('Straight')
            Steering_Landmarks_ids.append(0)
        elif val > turning_threshold:
            Steering_Landmarks.append('Right')
            Steering_Landmarks_ids.append(1)
        elif val < -turning_threshold:
            Steering_Landmarks.append('Left')
            Steering_Landmarks_ids.append(2)

    return Steering_Landmarks, Steering_Landmarks_ids


def writing_arff_file(data, combination, location):
    df = pd.DataFrame(data)

    data_list = []
    for i in df.index:
        val_list = []
        for col in df.columns:
            val_list.append(df[col][i])
        data_list.append(val_list)

    with open(
            "results/" + location + "/Abstraction_results/" + "abstract_inputs_clustering_" + location + "_" + combination + ".arff",
            "w") as file:
        file.write("@relation AUDI")
        file.write("\n")
        file.write("@attribute timestamps numeric")
        file.write("\n")
        file.write("@attribute acc_pedal numeric")
        file.write("\n")
        file.write("@attribute brake numeric")
        file.write("\n")
        file.write("@attribute steering numeric")
        file.write("\n")
        file.write("@attribute Speed numeric")
        file.write("\n")
        file.write("@attribute accelerator_pedal_ids numeric")
        file.write("\n")
        file.write("@attribute brake_pressure_ids numeric")
        file.write("\n")
        file.write("@attribute steering_angle_calculated_ids numeric")
        file.write("\n")
        file.write("@attribute vehicle_speed_ids numeric")
        file.write("\n")
        file.write("@data")
        file.write("\n")
        delimiter = ","
        for idx in range(len(data_list)):
            row = data_list[idx]
            # print(row)
            for j in range(len(row) - 1):
                file.write(str(row[j]) + delimiter)
            file.write(str(row[len(row) - 1]))
            file.write("\n")
        file.close()


def Clustering(data, combination, location, loader, method, nb_cluster):
    df = pd.DataFrame(data)
    clustering_input_data = loader.load_file(
        "results/" + location + "/Abstraction_results/" + "abstract_inputs_clustering_" + location + "_" + combination + ".arff")

    train_data_type = []
    removeATT = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first-5"])
    remove_duplicates = Filter(classname="weka.filters.unsupervised.instance.RemoveDuplicates")
    removeATT.inputformat(clustering_input_data)
    filtered_att = removeATT.filter(clustering_input_data)

    remove_duplicates.inputformat(filtered_att)
    filtered_redundancy = remove_duplicates.filter(filtered_att)
    data_size = filtered_redundancy.num_instances

    if method == "Simple K-means":
        print("Simple K-means with k = " + str(nb_cluster) + " using " + combination)
        train_data = filtered_redundancy
        df_metrics = pd.DataFrame(filtered_att, columns=filtered_att.attribute_names())

    elif method == "K-means with Auto-encoder":
        print("Auto-encoder with k = " + str(nb_cluster) + " using " + combination)
        Autoencoder = Filter(classname="weka.filters.unsupervised.attribute.MLPAutoencoder")
        Autoencoder.inputformat(filtered_redundancy)
        evaluation_dataEncoder = Autoencoder.filter(filtered_att)
        AutoE_filtered = Autoencoder.filter(filtered_redundancy)
        train_data = AutoE_filtered
        df_metrics = pd.DataFrame(evaluation_dataEncoder, columns=evaluation_dataEncoder.attribute_names())

    start_time = time.time()  # Start the timer

    clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", str(nb_cluster)])
    clusterer.build_clusterer(train_data)
    print(clusterer)
    evaluation = ClusterEvaluation()
    evaluation.set_model(clusterer)
    evaluation.test_model(train_data)

    print(evaluation)
    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time
    print("Time taken to build the model: {:.4f} seconds".format(elapsed_time))

    df_att = pd.DataFrame(filtered_att, columns=filtered_att.attribute_names())
    column_names = df_att.columns
    df_redundant = pd.DataFrame(filtered_redundancy, columns=filtered_redundancy.attribute_names())

    evaluation_df = pd.DataFrame({'cluster_assignments': evaluation.cluster_assignments})
    new_cluster_labels = []

    # Iterate over the rows of 'filtered_att' dataframe
    for index, att_row in df_att.iterrows():
        # Find matching instances in 'filtered_redundancy' dataframe based on column values
        conditions = df_redundant[column_names[0]] == att_row[column_names[0]]
        for col in column_names[1:]:
            conditions &= df_redundant[col] == att_row[col]

        match = df_redundant[conditions]

        if not match.empty:
            # Get the cluster assignment from 'evaluation_df' based on the index of the matched instance
            cluster_assignment = evaluation_df.loc[match.index[0], 'cluster_assignments']
            new_cluster_labels.append(cluster_assignment)

    df['cluster labels'] = new_cluster_labels

    # Generation graphs for the dataset "Gaimersheim
    if location == "Gaimersheim" and method == "K-means with Auto-encoder":
        print("Generating graphs for k-mean with autoencoders")
        print("nb of cluster for geenrating graphs", nb_cluster)
        print("new cluster_label for generating graphs", len(new_cluster_labels))
        find_sequences(new_cluster_labels, location, nb_cluster, method, combination)

    groups = df.groupby(df['cluster labels'])
    plt.figure(figsize=(15, 10))
    for name, group in groups:
        plt.plot(group['Timestamps'], group['Speed_values'], marker='o', linestyle='', markersize=2,
                 label=name)
    plt.legend(loc='upper left')
    plt.title('Abstraction approach using ' + method + ': Clustered vehicle speed' + ' using ' + combination)
    plt.show()

    plt.figure(figsize=(15, 10))
    for name, group in groups:
        plt.plot(group['Timestamps'], group['Steering_values'], marker='o', linestyle='', markersize=2,
                 label=name)
    plt.legend(loc='upper left')
    plt.title('Abstraction approach using ' + method + ': Clustered steering angle' + ' using ' + combination)
    plt.show()

    ### computing clustering validation metrics
    df_pcorr = df.drop(
        ['Acceleration_Landmarks_ids',
         'Brake_Landmarks_ids',
         'Steering_Landmarks_ids',
         'Speed_Landmarks_ids', 'cluster labels',
         'Timestamps'], axis=1)

    # Plot_clusters_episodes(method, df_pcorr, new_cluster_labels, nb_cluster, combination)

    DB_index = davies_bouldin_score(df_metrics, new_cluster_labels)
    CH_index = metrics.calinski_harabasz_score(df_metrics, new_cluster_labels)
    PCORR = Compute_PCORR_score(method, df_pcorr, new_cluster_labels, nb_cluster)
    silhouette_avg = silhouette_score(df_metrics, new_cluster_labels)
    print("Combination:", combination, "For k =", nb_cluster, ": AVG silhouette_score  = ",
          round(silhouette_avg, 4), "PCORR = ",
          PCORR, "DB index = ", round(DB_index, 4),
          "and CH index =", int(CH_index))

    return PCORR, silhouette_avg, DB_index, CH_index, elapsed_time, data_size


def Plot_clusters_episodes(method, data, cluster_assignments, nb_clusters, combination):
    for cluster in range(0, nb_clusters):
        fig = plt.figure(figsize=(12, 10))
        list_avg_pcorr_att = []
        for col, num in zip(data.columns, range(1, 5)):
            List_episodes = []
            Ep_val = []
            for inst in range(len(cluster_assignments) - 1):
                if cluster_assignments[inst] == cluster:
                    Ep_val.append(data[col][inst])
                    if cluster_assignments[inst + 1] != cluster and len(Ep_val) > 20:
                        List_episodes.append(Ep_val)
                        Ep_val = []

            if len(List_episodes) > 0:
                ax = fig.add_subplot(2, 2, num)
                df = pd.DataFrame(List_episodes)

                df_t = df.T
                ax.plot(df_t)
                ax.set_title(col)

        fig.suptitle("Method: " + method + " with k = " +
                     str(nb_clusters) + "Combination: " + combination + " Cluster " + str(cluster))

        plt.show()


def Compute_Mean_values(val_list):
    window = 200
    avg_list = []

    for i in range(len(val_list)):
        if i >= window + 1:
            idx_low = i - window
        else:
            idx_low = 0

        if i < len(val_list) - window:
            idx_up = i + window + 1
        else:
            idx_up = len(val_list)

        y = val_list[idx_low:idx_up]
        y_mean = np.mean(y)

        avg_list.append(y_mean)
    return avg_list


def Compute_PCORR_score(method, data, cluster_assignments, nb_clusters):
    list_avg_pcorr = []
    for cluster in range(0, nb_clusters):
        fig = plt.figure(figsize=(12, 10))
        list_avg_pcorr_att = []
        for col, num in zip(data.columns, range(1, 5)):
            List_episodes = []
            Ep_val = []
            for inst in range(len(cluster_assignments) - 1):
                if cluster_assignments[inst] == cluster:
                    Ep_val.append(data[col][inst])
                    if cluster_assignments[inst + 1] != cluster and len(Ep_val) > 20:
                        List_episodes.append(Ep_val)
                        Ep_val = []
            nb_empty_cluster = 0
            if len(List_episodes) == 0:
                list_avg_pcorr_att.append('NaN')
                nb_empty_cluster += 1
                print('Empty clusters', nb_empty_cluster)

            elif len(List_episodes) > 1:
                ax = fig.add_subplot(2, 2, num)
                df = pd.DataFrame(List_episodes)

                df_t = df.T
                ax.plot(df_t)
                ax.set_title(col)

                corr = df_t.corr(method='pearson', min_periods=1)
                List_correlated_Eps = []
                List_i = []
                for i in range(0, len(corr)):
                    nbr_correlated_ep_perLine = 0
                    for j in range(0, len(corr[i])):
                        if i < j:
                            if corr[i][j] >= 0.5:
                                if j not in List_i:
                                    List_i.append(j)
                                    List_correlated_Eps.append(df_t[j])
                                    nbr_correlated_ep_perLine += 1

                total_correlated_ep_per_att = len(List_i)
                pcorr_per_att = (total_correlated_ep_per_att / len(List_episodes)) * 100

                list_avg_pcorr_att.append(round(pcorr_per_att, 2))
                # print("percentage of  pearson correlation_per_att", round(pcorr_per_att, 2))
        fig.suptitle("Method: " + method + " with k = " +
                     str(nb_clusters) + " in Cluster " + str(cluster))

        plt.show()

        if len(list_avg_pcorr_att) > 0 and list_avg_pcorr_att.__contains__('NaN') == False:
            list_avg_pcorr.append(round(np.mean(list_avg_pcorr_att), 2))
    avg_pcorr_final = round(np.mean(list_avg_pcorr), 2)

    return avg_pcorr_final


def find_sequences(new_cluster_labels, location, nb_clusters, method, combination):
    # driving scenarios
    S1 = "Braking before crossing or turning"
    S2 = "Accelerating after crossing or turning"
    S3 = "Car turning"
    S4 = "Approaching a crossroad or an obstacle"
    S5 = "Car stopping at red traffic light or traffic jam"
    S6 = "Car accelerating after stopping at red traffic light or traffic jam"
    S7 = "Driving straight on a highway"
    S8 = "Car in a roundabout"
    S9 = "Car braking"
    S10 = "Car accelerating"
    S11 = "Car driving at a constant speed"
    labels_order = []
    if nb_clusters == 4:
        if combination == 'Speed_model1 and Steering_model1':
            print(method + " using " + combination)
            labels_order = [S4, S2, S3, S1]

        elif combination == 'Speed_model1 and Steering_model2':
            print(method + " using " + combination)
            labels_order = [S1, S4, S9, S3]

        elif combination == 'Speed_model2 and Steering_model1':
            print(method + " using " + combination)
            labels_order = [S4, S3, S2, S1]

        elif combination == 'Speed_model2 and Steering_model2':
            print(method + " using " + combination)
            labels_order = [S1, S3, S2, S4]

        elif combination == 'Speed_model3 and Steering_model1':
            print(method + " using " + combination)
            labels_order = [S4, S3, S1, S2]

        elif combination == 'Speed_model3 and Steering_model2':
            print(method + " using " + combination)
            labels_order = [S2, S1, S3, S4]

    if nb_clusters == 8:
        if combination == 'Speed_model1 and Steering_model1':
            print(method + " using " + combination)
            labels_order = [S11, S3, S9, S10, S1, S4, S7, S9]
        elif combination == 'Speed_model1 and Steering_model2':
            print(method + " using " + combination)
            labels_order = [S2, S4, S7, S8, S3, S9, S1, S11]

        elif combination == 'Speed_model2 and Steering_model1':
            print(method + " using " + combination)
            labels_order = [S9, S11, S2, S4, S2, S11, S7, S1]

        elif combination == 'Speed_model2 and Steering_model2':
            print(method + " using " + combination)
            labels_order = [S9, S1, S8, S9, S3, S4, S2, S4]

        elif combination == 'Speed_model3 and Steering_model1':
            print(method + " using " + combination)
            labels_order = [S1, S4, S11, S3, S11, S7, S7, S4]

        elif combination == 'Speed_model3 and Steering_model2':
            print(method + " using " + combination)
            labels_order = [S4, S3, S7, S2, S11, S1, S11, S2]

    clusters_labels = []
    for cluster in range(0, nb_clusters):
        clusters_labels.append([cluster, labels_order[cluster]])

    sequence_actions = []
    cluster_ids = []

    for inst in range(len(new_cluster_labels) - 1):
        cluster_ids.append(new_cluster_labels[inst])
        if new_cluster_labels[inst + 1] != new_cluster_labels[inst] and len(cluster_ids) > 20:
            # print(cluster_ids)
            for label in clusters_labels:
                if cluster_ids[0] == label[0]:
                    action = label[1]
                    # print(action)
                    sequence_actions.append(action)
                    cluster_ids = []
                    break
    print("number of episodes", len(sequence_actions))
    compute_transitions(sequence_actions, labels_order, location, nb_clusters, method, combination)


def compute_transitions(sequence_actions, labels_order, location, nb_clusters, method, combination):
    with open('results/' + location + '/Abstraction_results/'
                                      '/Transitions_dict_' + location + '_K' + str(nb_clusters) + '.json',
              'w') as f:

        transitions = {}
        for s_in in labels_order:
            transitions[s_in] = {}
            for s_out in labels_order:
                transitions[s_in][s_out] = {'nbr_edges': 0}

        for a in range(0, len(sequence_actions) - 1):
            for state in labels_order:
                if sequence_actions[a + 1] == state:
                    transitions[sequence_actions[a]][state]['nbr_edges'] += 1

        # print("final automata status ")
        # print(transitions)
        json.dump(transitions, f, sort_keys=True, indent=4)
        create_automata_json_input(sequence_actions, labels_order, transitions, location, nb_clusters, method,
                                   combination)


def create_automata_json_input(sequence_actions, labels_order, transitions, location, nb_clusters, method, combination):
    transitions_list = []
    alphabet = []
    for key in transitions:
        for s_out in labels_order:
            if transitions[key][s_out]['nbr_edges'] > 0:
                edge_name = str(transitions[key][s_out]['nbr_edges'])
                T = [key, edge_name, s_out]
                alphabet.append(edge_name)
                transitions_list.append(T)
    # print("transition list ", transitions_list)
    print("sequence of actions", sequence_actions)
    with open('results/' + location + '/Abstraction_results' +
              '/Automata_' + location + '_' + method + '_using_' + combination + '_K' + str(nb_clusters) + '.json',
              'w') as f:
        print("initial node", sequence_actions[0])
        digraph = {"alphabet": alphabet, "states": [], "initial_state": sequence_actions[0], "accepting_states": [],
                   "transitions": []}
        for s in labels_order:
            digraph["states"].append(s)

        digraph["transitions"] = transitions_list

        print(digraph)
        json.dump(digraph, f, sort_keys=True, indent=4)

    dfa_example = PySimpleAutomata.automata_IO.dfa_json_importer(
        'results/' + location + '/Abstraction_results' +
        '/Automata_' + location + '_' + method + '_using_' + combination + '_K' + str(nb_clusters) + '.json')

    automata_IO.dfa_to_dot(dfa_example, 'automata_graph_' + location + '_K' + str(nb_clusters),
                           'results/' + location + '/Abstraction_results' + '_' + method + '_using_' + combination + '_K' + str(
                               nb_clusters) +
                           '/')

    automata_IO.dfa_to_json(dfa_example, 'automata_graph_' + location + '_K' + str(nb_clusters),
                            'results/' + location + '/Abstraction_results' + '_' + method + '_using_' + combination + '_K' + str(
                                nb_clusters) +
                            '/')


def Plot_Histograms():
    file_path = "results/Data_abstraction(difference_results).xlsx"
    df = pd.read_excel(file_path)
    print(df.columns)

    df = df.fillna(method='ffill', axis=0)
    print(df)
    grouped_df = df.groupby(["Method", "K"])
    k = df["K"].unique()
    method = df["Method"].unique()
    metric = ["PCORR", "Avg_S"]

    for idx, metric_val in enumerate(metric):
        if metric_val == "PCORR":
            w = 1.75
        elif metric_val == "Avg_S":
            w = 0.05
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

        for i, m in enumerate(method):
            for j, k_val in enumerate(k):
                axs[i, j].set_title(str(m) + " using " + str(k_val))
                sub_df = grouped_df.get_group((m, k_val))

                sub_df[f"{metric_val}_abstract"].plot(kind="hist", ax=axs[i, j], legend=True, zorder=2, rwidth=0.9,
                                                      alpha=0.7,
                                                      bins=np.arange(min(sub_df[f"{metric_val}_abstract"]),
                                                                     max(sub_df[f"{metric_val}_abstract"]) + w, w))
                sub_df[f"{metric_val}_numeric"].plot(kind="hist", ax=axs[i, j], legend=True, zorder=2, rwidth=0.9,
                                                     alpha=0.7,
                                                     bins=np.arange(min(sub_df[f"{metric_val}_numeric"]),
                                                                    max(sub_df[f"{metric_val}_numeric"]) + w, w))

        plt.suptitle(f" Clustering Evaluation Metric: {metric_val}")

    plt.legend()

    for idx, metric_val in enumerate(metric):
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

        for i, m in enumerate(method):
            for j, k_val in enumerate(k):
                axs[i, j].set_title(str(m) + " using " + str(k_val))
                sub_df = grouped_df.get_group((m, k_val))

                sub_df[f"{metric_val}_difference"].plot(kind="hist", ax=axs[i, j], legend=True)

        plt.suptitle(f" Clustering Evaluation Metric: {metric_val}")

    plt.legend()
    plt.show()
    exit()


def main():
    loader = Loader("weka.core.converters.ArffLoader")

    datasets = ["Gaimersheim", "Munich", "Ingolstadt"]

    for location in datasets:
        path_to_clean_data = os.path.join(
            "data/Clustering_input_data/",
            location,
            "test_data_cleaned.arff")
        path_to_filtered_data = os.path.join(
            "data/Clustering_input_data/",
            location,
            "test_data_Filtered.arff")

        clean_data = loader.load_file(path_to_clean_data)
        input_data = loader.load_file(path_to_filtered_data)

        path_to_df_json = os.path.join('data/',
                                       'Interpolated_data/', 'data_' + location + '.json')
        print(path_to_df_json)
        df_json = pd.read_json(path_to_df_json)
        steering_angle_cal_sign = df_json["steering_angle_calculated_sign"]["values"]

        steering_sign = []
        for val in steering_angle_cal_sign:
            steering_sign.append(val[1])

        Speed_val_list = []
        TS_list = []
        Brake_val_list = []
        Acc_val_list = []
        Steering_val_list = []

        for inst in range(0, clean_data.num_instances):
            TS_list.append(clean_data.get_instance(inst).get_value(0))
            Acc_val_list.append(input_data.get_instance(inst).get_value(0))
            Brake_val_list.append(input_data.get_instance(inst).get_value(1))
            Steering_val_list.append(input_data.get_instance(inst).get_value(2))
            Speed_val_list.append(input_data.get_instance(inst).get_value(3))

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        axes[0, 0].hist(Acc_val_list, bins=20, rwidth=0.9)
        axes[0, 1].hist(Brake_val_list, bins=20, rwidth=0.9)
        axes[1, 0].hist(Steering_val_list, bins=20, rwidth=0.9)
        axes[1, 1].hist(Speed_val_list, bins=20, rwidth=0.9)

        axes[0, 0].title.set_text("Acceleration pedal")
        axes[0, 0].set_xlabel("values")
        axes[0, 1].title.set_text("Brake pressure ")
        axes[0, 1].set_xlabel("values")
        axes[1, 0].title.set_text("Steering angle")
        axes[1, 0].set_xlabel("values")
        axes[1, 1].title.set_text("Vehicle speed")
        axes[1, 1].set_xlabel("values")

        plt.figure(figsize=(15, 10))
        plt.plot(TS_list, steering_sign)
        plt.title('steering_angle_cal_sign')

        print('**Acceleration pedal abstraction**')
        Acc_avg_list = Compute_Mean_values(Acc_val_list)
        Acc_Landmarks = []
        Acc_Landmarks_ids = []

        for mean in Acc_avg_list:
            if mean <= 1:
                Acc_Landmarks.append('Off')
                Acc_Landmarks_ids.append(0)
            else:
                Acc_Landmarks.append('On')
                Acc_Landmarks_ids.append(1)

        data_Acc = {'timestamps': TS_list,
                    'avg_values': Acc_avg_list,
                    'Landmarks': Acc_Landmarks,
                    }

        df_Acc = pd.DataFrame(data_Acc)
        groups = df_Acc.groupby(df_Acc['Landmarks'])
        plt.figure(figsize=(10, 6))
        for name, group in groups:
            plt.plot(group['timestamps'], group['avg_values'], marker='o', linestyle='', markersize=2,
                     label=name)
        plt.legend(loc='upper left')
        plt.title('Acceleration Pedal Landmarks')

        print("**Brake Pressure abstraction**")
        Brake_avg_list = Compute_Mean_values(Brake_val_list)

        Brake_Landmarks = []
        Brake_Landmarks_ids = []
        for mean in Brake_avg_list:
            if mean <= 0:
                Brake_Landmarks.append('Off')
                Brake_Landmarks_ids.append(0)
            else:
                Brake_Landmarks.append('On')
                Brake_Landmarks_ids.append(1)

        data_brake = {'timestamps': TS_list,
                      'avg_values': Brake_avg_list,
                      'Landmarks': Brake_Landmarks,
                      }
        df_brake = pd.DataFrame(data_brake)
        groups = df_brake.groupby(df_brake['Landmarks'])
        plt.figure(figsize=(10, 6))
        for name, group in groups:
            plt.plot(group['timestamps'], group['avg_values'], marker='o', linestyle='', markersize=2,
                     label=name)
        plt.legend(loc='upper left')
        plt.title('Braking pressure Landmarks')

        print("**Steering angle abstraction**")

        Steering_avg_list = Compute_Mean_values(Steering_val_list)
        plt.figure(figsize=(15, 10))
        plt.subplot(3, 1, 1)
        plt.plot(TS_list, Steering_val_list)
        plt.title('Steering angle values')
        plt.subplot(3, 1, 2)
        plt.plot(TS_list, Steering_avg_list)
        plt.title('Steering angle Means values')
        plt.subplot(3, 1, 3)
        plt.plot(steering_sign)
        plt.title('Steering sign ')

        steering_with_sign__val = []

        for value, direction in zip(Steering_val_list, steering_sign):
            if direction < 1:
                tmp_val = value * (-1)
            elif direction == 1:
                tmp_val = value

            steering_with_sign__val.append(tmp_val)

        Steering_avg_list_M2 = Compute_Mean_values(steering_with_sign__val)

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(TS_list, steering_with_sign__val)
        plt.title('steering_with_sign__val')
        plt.subplot(2, 1, 2)
        plt.plot(TS_list, Steering_avg_list_M2)
        plt.title('Steering_avg_list_M2')

        print("**Vehicle speed abstraction**")

        speed_avg_list = Compute_Mean_values(Speed_val_list)

        Speed_abstractions = ["Speed_model1", "Speed_model2", "Speed_model3"]
        Steering_abstractions = ["Steering_model1", "Steering_model2"]
        # Iterate over the list and call the functions dynamically

        methods = ["Simple K-means", "K-means with Auto-encoder"]
        for method in methods:
            range_n_clusters = [4, 8]
            for nb_cluster in range_n_clusters:

                with open('results/' + location + '/Abstraction_results' +
                          '/Abstraction_results_' + location + '_' + method + '_K' + str(nb_cluster) + '.txt',
                          'w') as file:
                    results = {'Combination': [],
                               'avg PCORR': [],
                               'avg Silhouette_Score': [],
                               'DB_index': [],
                               'CH_index': [],
                               'Clustering_time(s)': [],
                               'Data_size': []}
                    for i in Speed_abstractions:
                        for j in Steering_abstractions:
                            print(i, j)
                            Speed_Landmarks, Speed_Landmarks_ids = eval(i)(speed_avg_list)
                            if j == 'Steering_model1':
                                Steering_Landmarks, Steering_Landmarks_ids = eval(j)(
                                    Steering_avg_list)  # Call the function using eval()
                                data_Steering = {'timestamps': TS_list,
                                                 'avg_values': Steering_avg_list,
                                                 'Landmarks': Steering_Landmarks,
                                                 }

                            if j == 'Steering_model2':
                                Steering_Landmarks, Steering_Landmarks_ids = eval(j)(Steering_avg_list_M2)
                                data_Steering = {'timestamps': TS_list,
                                                 'avg_values': Steering_avg_list_M2,
                                                 'Landmarks': Steering_Landmarks,
                                                 }

                            data_speed = {'timestamps': TS_list,
                                          'avg_values': speed_avg_list,
                                          'Landmarks': Speed_Landmarks,
                                          }

                            df_speed = pd.DataFrame(data_speed)
                            groups = df_speed.groupby(df_speed['Landmarks'])
                            plt.figure(figsize=(10, 6))
                            for name, group in groups:
                                plt.plot(group['timestamps'], group['avg_values'], marker='o', linestyle='',
                                         markersize=2,
                                         label=name)
                            plt.legend(loc='upper left')
                            plt.title('Speed Landmarks using ' + i)
                            plt.show()
                            print("done abstraction of the data_speed values for speed ")
                            df_Steering = pd.DataFrame(data_Steering)
                            groups = df_Steering.groupby(df_Steering['Landmarks'])
                            plt.figure(figsize=(10, 6))
                            for name, group in groups:
                                plt.plot(group['timestamps'], group['avg_values'], marker='o', linestyle='',
                                         markersize=2,
                                         label=name)
                            plt.legend(loc='upper left')
                            plt.title('Steering angle Landmarks using ' + j)
                            plt.show()

                            data = {'Timestamps': TS_list,
                                    'Acc_values': Acc_val_list, 'Brake_values': Brake_val_list,
                                    'Steering_values': Steering_val_list,
                                    'Speed_values': Speed_val_list,
                                    'Acceleration_Landmarks_ids': Acc_Landmarks_ids,
                                    'Brake_Landmarks_ids': Brake_Landmarks_ids,
                                    'Steering_Landmarks_ids': Steering_Landmarks_ids,
                                    'Speed_Landmarks_ids': Speed_Landmarks_ids}
                            combination = i + ' and ' + j
                            writing_arff_file(data, combination, location)

                            PCORR, silhouette_avg, DB_index, CH_index, elapsed_time, data_size = Clustering(data,
                                                                                                            combination,
                                                                                                            location,
                                                                                                            loader,
                                                                                                            method,
                                                                                                            nb_cluster)
                            results['Combination'].append(combination)
                            results['avg PCORR'].append(PCORR)
                            results['avg Silhouette_Score'].append(round(silhouette_avg, 4))
                            results['DB_index'].append(round(DB_index, 4))
                            results['CH_index'].append(int(CH_index) / 1000)
                            results['Clustering_time(s)'].append(round(elapsed_time, 4))
                            results['Data_size'].append(data_size)

                            df_results = pd.DataFrame(results)
                            print(
                                df_results.to_latex(index=False,
                                                    caption=method + ' using ' + str(nb_cluster) + ' in ' + location))
                    print(
                        df_results.to_latex(index=False,
                                            caption=method + ' using ' + str(nb_cluster) + ' in ' + location))
                    df_string = df_results.to_string(header=False, index=False)
                    file.write(df_string)


if __name__ == "__main__":
    try:
        jvm.start(system_cp=True, packages=True, max_heap_size="512m")
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
