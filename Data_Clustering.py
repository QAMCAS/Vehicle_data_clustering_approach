# Using Weka clusterer - requires python-weka-wrapper3 later than 0.2.4
import traceback

import PySimpleAutomata
import weka.core.jvm as jvm
import os
import json

from PySimpleAutomata import automata_IO
from scipy import interpolate
from weka.core import packages
from weka.core.dataset import Instances
from weka.clusterers import Clusterer
from weka.clusterers import ClusterEvaluation
from weka.core.converters import Loader
from weka.filters import Filter, MultiFilter
import weka.plot as plot

if plot.matplotlib_available:
    import matplotlib.pyplot as plt
from matplotlib import pyplot
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# logging setup
logger = logging.getLogger(__name__)


def main(location):
    loader = Loader("weka.core.converters.ArffLoader")
    path_to_clean_data = os.path.join("data/Clustering_input_data/", location, "test_data_cleaned.arff")
    path_to_filtered_data = os.path.join("data/Clustering_input_data/", location, "test_data_Filtered.arff")
    clean_data = loader.load_file(path_to_clean_data)
    input_data = loader.load_file(path_to_filtered_data)

    # Preparing results directory
    path_folder_loc1 = os.path.join('results/Gaimersheim')
    if not os.path.exists(path_folder_loc1):
        os.makedirs(path_folder_loc1)
    path_folder_loc2 = os.path.join('results/Munich')
    if not os.path.exists(path_folder_loc2):
        os.makedirs(path_folder_loc2)
    path_folder_loc3 = os.path.join('results/Ingolstadt')
    if not os.path.exists(path_folder_loc3):
        os.makedirs(path_folder_loc3)

    List_attributes = []
    for att in range(0, clean_data.num_attributes):
        List_attributes.append(clean_data.attribute(att))
    # print(List_attributes)

    List_of_clusterers = ['SimpleKMeans', 'EM', 'Canopy', 'SelfOrganizingMap'
                          ]  # for 'SelfOrganizingMap' you need to remove nb_clusters
    range_n_clusters = [4, 8]  # , 5, 6, 7, 8] # uncomment to increase number of clusters
    for nb_clusters in range_n_clusters:
        # comment out if you want to try the other algorithms
        '''
        for algorithm in List_of_clusterers:

            if algorithm != 'SelfOrganizingMap':
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
                
                clusterer_name = str(clusterer.classname)
                save_cluster_assignments(clusterer_name, evaluation, location, nb_clusters)

                print("1. Creating JSON files for each cluster  for ", clusterer.classname, "using ", nb_clusters)
                json_file_name = str(clusterer.classname) + '_K_' + str(nb_clusters)
                create_JSON_files(location, clean_data, clusterer, nb_clusters, evaluation, json_file_name)

                print("2. Computing Average Pearson Coefficient for ", str(clusterer.classname), "using",
                      str(nb_clusters),
                      "clusters")
                PCORR_file_name = "PCORR_" + str(clusterer.classname) + "_K_" + str(nb_clusters)
                AVG_Percentage_Pearson_correlation(location, input_data, clean_data, clusterer, evaluation,
                                                   PCORR_file_name)

            if algorithm == 'SelfOrganizingMap' and nb_clusters == 4:
                clusterer = Clusterer(classname="weka.clusterers." + str(algorithm))
                clusterer.build_clusterer(input_data)
                print(clusterer)
                evaluation = ClusterEvaluation()
                evaluation.set_model(clusterer)
                evaluation.test_model(input_data)
                print("evaluation;" + str(evaluation.cluster_results))
                print("# clusters: " + str(evaluation.num_clusters))
                print("log likelihood: " + str(evaluation.log_likelihood))
                print("cluster assignments:\n" + str(evaluation.cluster_assignments))
                
                clusterer_name = str(clusterer.classname)
                save_cluster_assignments(clusterer_name, evaluation, location, nb_clusters)
                print("1. Creating JSON files for each cluster  for ", clusterer.classname, "using ", nb_clusters)
                create_JSON_files(location, clean_data, clusterer, nb_clusters, evaluation, PCORR_file_name)
                print("2. Computing Average Pearson Coefficient for ", str(clusterer.classname), "using",
                      str(nb_clusters),
                      "clusters")
                PCORR_file_name = "PCORR_" + str(clusterer.classname) + "_K_" + str(nb_clusters)
                AVG_Percentage_Pearson_correlation(location, input_data, clean_data, clusterer, evaluation,
                                                   PCORR_file_name)
        '''
        # clustering Using SimpkeKMeans with autoencoder preprocessing
        print("**** Clustering using auto-encoder ****")
        Autoencoder = Filter(classname="weka.filters.unsupervised.attribute.MLPAutoencoder")
        Autoencoder.inputformat(input_data)
        filtered = Autoencoder.filter(input_data)  # data filtered with  autoencoder
        clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", str(nb_clusters)])
        clusterer.build_clusterer(filtered)
        print(clusterer)
        evaluation = ClusterEvaluation()
        evaluation.set_model(clusterer)
        evaluation.test_model(filtered)

        # saving clusters assignments if not saved
        clusterer_name = str(clusterer.classname) + "with_Autoencoder"
        save_cluster_assignments(clusterer_name, evaluation, location, nb_clusters)

        print("1. Creating JSON files for each cluster  for ", clusterer.classname, " with_Autoencoder using ",
              nb_clusters)
        json_file_name = clusterer_name + "_K_" + str(nb_clusters)
        create_JSON_files(location, clean_data, clusterer, nb_clusters, evaluation, json_file_name)
        print("2.Computing Average Pearson Coefficient for ", str(clusterer.classname), "with Autoencoder using",
              str(nb_clusters), "clusters")
        PCORR_file_name = "PCORR_" + clusterer_name + "_K_" + str(nb_clusters)
        AVG_Percentage_Pearson_correlation(location, input_data, clean_data, clusterer, evaluation, PCORR_file_name)

        print("Creating automata models and TC concretization ")
        find_sequences(clusterer, clusterer_name, evaluation, location, nb_clusters)


# write extracted driving scenarios in json files
def create_JSON_files(location, clean_data, clusterer, nb_clusters, evaluation, json_file_name):
    cluster_data_path = os.path.join('results', location)
    if not os.path.exists(os.path.join(cluster_data_path, "Driving_scenarios")):
        os.makedirs(os.path.join(cluster_data_path, "Driving_scenarios"))
    if not os.path.exists(os.path.join(cluster_data_path, "Driving_scenarios", 'json_files')):
        os.makedirs(os.path.join(cluster_data_path, "Driving_scenarios", 'json_files'))
    for cluster in range(clusterer.number_of_clusters):
        with open(
                os.path.join(cluster_data_path, 'driving_scenarios/', 'json_files', json_file_name + '_scenario_' + str(
                    cluster) + '.json'),
                'w') as f:

            episodes_list = []
            # initiate val dict
            val_dict = {}
            for att in clean_data.attribute_names():
                val_dict[att] = {'values': []}

            for inst in range(0, len(evaluation.cluster_assignments) - 1):
                if evaluation.cluster_assignments[inst] == cluster:
                    for i in range(len(clean_data.attribute_names())):
                        att_name = clean_data.attribute_names()[i]
                        inst_att_value = clean_data.get_instance(inst).get_value(i)
                        val_dict[att_name]['values'].append(inst_att_value)
                    if evaluation.cluster_assignments[inst + 1] != cluster and len(
                            val_dict['timestamps']['values']) >= 150:
                        episodes_list.append(val_dict)
                        val_dict = {}
                        for att in clean_data.attribute_names():
                            val_dict[att] = {'values': []}

            ep_dict = {'Episodes': []}
            for j in range(0, len(episodes_list)):
                ep_dict['Episodes'].append({'id_{}'.format(j): {'Sensors': episodes_list[j]}})

            json.dump(ep_dict, f, sort_keys=True, indent=4)

    print("Writing json files  for ", clusterer.classname, "using ", nb_clusters, " clusters is done")


# compute percentage of Pearson correlation in each cluster save PCorr results in a txt file
def AVG_Percentage_Pearson_correlation(location, data_filtered, clean_data, clusterer, evaluation, PCORR_file_name):
    PCORR_data_path = os.path.join('results', location)
    if not os.path.exists(os.path.join(PCORR_data_path, "Pearson_correlation_results")):
        os.makedirs(os.path.join(PCORR_data_path, "Pearson_correlation_results"))

    file_correlation = open(
        "results/" + location + "/Pearson_correlation_results/" + PCORR_file_name + ".txt",
        "w")
    delimiter = ";"
    file_correlation.write(
        "Cluster" + delimiter + "Nb of total episodes" + delimiter + "Percentage of Pearson correlation")
    file_correlation.write("\n")
    sum_pcorr_cluster = 0
    for cluster in range(0, clusterer.number_of_clusters):
        fig = plt.figure(figsize=(8, 6))
        Sum_pcorr = 0
        file_correlation.write(str(cluster) + delimiter)
        for (att, num) in zip(range(0, data_filtered.num_attributes), range(1, 5)):
            List_episodes = []
            List_timestamps = []
            val_list = []
            for inst in range(len(evaluation.cluster_assignments) - 1):
                if evaluation.cluster_assignments[inst] == cluster:
                    # print("ok3")
                    val_list.append(data_filtered.get_instance(inst).get_value(att))
                    List_timestamps.append(clean_data.get_instance(inst).get_value(att))
                    if evaluation.cluster_assignments[inst + 1] != cluster and len(val_list) > 20:
                        List_episodes.append(val_list)
                        val_list = []
            # print(" number of episodes in cluster ", cluster, "for ", str(data_filtered.attribute(att)), " ",
            # len(List_episodes))

            if len(List_episodes) > 0:
                df = pd.DataFrame(List_episodes)
                df_t = df.T
                df_t.dropna()
                ax = fig.add_subplot(2, 2, num)
                fig.suptitle(clusterer.classname + "/Cluster " + str(cluster))
                ax.set_title(str(data_filtered.attribute_names()[att]))
                ax.plot(df_t)

                corr = df_t.corr(method='pearson', min_periods=1)
                List_i = []
                for i in range(0, len(corr)):
                    nbr_correlated_ep_perLine = 0
                    for j in range(0, len(corr[i])):
                        if i < j:
                            if corr[i][j] >= 0.5:
                                if j not in List_i:
                                    List_i.append(j)
                                    nbr_correlated_ep_perLine += 1

                total_correlated_ep_per_att = len(List_i)
                # print("total of correlated episodes per attribute", total_correlated_ep_per_att)
                pcorr_per_att = (total_correlated_ep_per_att / len(List_episodes)) * 100
                pcorr_per_att = round(pcorr_per_att, 2)
                # print("percentage of  pearson correlation_per_att", pcorr_per_att)
                Sum_pcorr = Sum_pcorr + pcorr_per_att

        avg_pcorr_cluster = round(Sum_pcorr / data_filtered.num_attributes, 2)
        sum_pcorr_cluster = sum_pcorr_cluster + avg_pcorr_cluster
        print("Average percentage of Pearson Coefficient per cluster", cluster, "=", avg_pcorr_cluster)
        file_correlation.write(
            str(len(List_episodes)) + delimiter + str(avg_pcorr_cluster))
        file_correlation.write("\n")
        plt.tight_layout
        # comment in to plot the episodes in each cluster
        # plt.show()
    avg_pcorr_final = round(sum_pcorr_cluster / clusterer.number_of_clusters, 2)
    print("Average Percentage of Pearson coefficient for", PCORR_file_name, "= ", avg_pcorr_final)
    file_correlation.write("Final Average:" + str(avg_pcorr_final))

    file_correlation.write("\n")
    file_correlation.close()


# extract episodes
def ExtractEpisodes(data_filtered, cluster, evaluation, att):
    List_episodes = []
    val_list = []
    for inst in range(len(evaluation.cluster_assignments) - 1):
        if evaluation.cluster_assignments[inst] == cluster:
            val_list.append(data_filtered.get_instance(inst).get_value(att))
            if evaluation.cluster_assignments[inst + 1] != cluster and len(val_list) > 20:
                List_episodes.append(val_list)
                val_list = []
    return List_episodes


# plots graphs of extracted episodes, probability distribution and Pearson correlation matrix for each attribute and
# save their files
def Plot_Episodes_ProbabilityDist_PearsonCorrMatrix(location, input_data, clean_data, clusterer, nb_clusters,
                                                    evaluation):
    if not os.path.exists(os.path.join("results/", location, "Plots")):
        os.makedirs(os.path.join("results/", location, "Plots"))
    for cluster in range(0, clusterer.number_of_clusters):
        if not os.path.exists(os.path.join("results/", location, "Plots/Cluster_" + str(cluster))):
            os.makedirs(os.path.join("results/", location, "Plots/Cluster_" + str(cluster)))

        path_to_file = "results/" + location + "/Plots/Cluster_" + str(cluster) + "/"

        for (att, num) in zip(range(0, input_data.num_attributes), range(1, 5)):
            List_episodes = []
            List_timestamps = []
            val_list = []
            for inst in range(len(evaluation.cluster_assignments) - 1):
                if evaluation.cluster_assignments[inst] == cluster:
                    val_list.append(input_data.get_instance(inst).get_value(att))
                    List_timestamps.append(clean_data.get_instance(inst).get_value(att))
                    if evaluation.cluster_assignments[inst + 1] != cluster and len(val_list) > 20:
                        List_episodes.append(val_list)
                        val_list = []

            if len(List_episodes) > 0:
                df = pd.DataFrame(List_episodes)
                df_t = df.T
                df_t.dropna()

                # Plot extracted episodes for each attribute
                plt.figure(figsize=(10, 20))
                df_t.plot(legend=True)
                plt.title("Episodes " + str(input_data.attribute_names()[att]) + " in cluster " + str(cluster))
                plt.savefig(path_to_file + "_K" + str(nb_clusters) + "_Episodes " + str(
                    input_data.attribute_names()[att]) + " in cluster " + str(cluster))
                # plt.show()

                # Plot episodes probability distribution  for each attribute
                df_t.plot.hist(density=True, legend=False, grid=False, bins=20, rwidth=0.9)
                plt.xlabel('Values')
                plt.ylabel('Probability density distribution')
                plt.grid(axis='x', alpha=0.75)
                plt.title("Distribution" + str(input_data.attribute_names()[att]) + " in cluster  " + str(cluster))
                plt.savefig(
                    path_to_file + "_K" + str(nb_clusters) + "_Distribution" + str(
                        input_data.attribute_names()[att]) + " in cluster  " + str(cluster))
                # plt.show()

                # Plot Pearson coefficient matrix for each attribute
                corr = df_t.corr(method='pearson', min_periods=1)
                mask = np.zeros_like(corr, dtype='bool')
                f, ax = pyplot.subplots(figsize=(12, 10))
                cmap = sns.diverging_palette(220, 10, as_cmap=True)
                sns.heatmap(corr, mask=mask, cmap=cmap,
                            square=True, annot=True, linewidths=.5, ax=ax)

                pyplot.title("Pearson Correlation matrix of the episodes  " + str(
                    input_data.attribute_names()[att]) + " in cluster  " + str(cluster))
                plt.savefig(
                    path_to_file + "_K" + str(nb_clusters) + "Pearson Correlation matrix of the episodes  " + str(
                        input_data.attribute_names()[att]) + " in cluster  " + str(cluster))


def Create_arff_outputfiles(data_cleaned, location, clusterer, evaluation, nb_clusters, List_attributes):
    # Writing extracted driving scenarios  in  arff file for weka use
    if not os.path.exists(os.path.join("results/", location, "Driving_scenarios", 'arff_files')):
        os.makedirs(os.path.join("results/", location, "Driving_scenarios", 'arff_files'))
    for cluster in range(0, clusterer.number_of_clusters):
        dataset_scenario = Instances.create_instances("scenario_" + str(cluster), List_attributes, 0)
        for att in range(0, data_cleaned.num_attributes):
            timestamps_list = []
            val_list = []
            stop = False
            for inst in range(len(evaluation.cluster_assignments)):
                if evaluation.cluster_assignments[inst] == cluster and stop == False:
                    val_list.append(data_cleaned.get_instance(inst).get_value(att))
                    timestamps_list.append(data_cleaned.get_instance(inst).get_value(0))
                    dataset_scenario.add_instance(data_cleaned.get_instance(inst))

                    if evaluation.cluster_assignments[inst + 1] != cluster and len(val_list) > 20:
                        stop = True

        file_scenario = open(
            "results/" + location + "/Driving_scenarios/arff_files/" + str(clusterer.classname) + "_K" + str(
                nb_clusters) + "_Scenario_" + str(
                cluster) + ".arff", "w")
        file_scenario.write(str(dataset_scenario))

    file_scenario = open(
        "Results/" + location + "/Driving_scenarios/arff_files/" + str(clusterer.classname) + "_Scenario_" + str(
            cluster) + ".arff", "w")
    file_scenario.write(str(dataset_scenario))

    file_scenario.close()
    print("Writing results in arff file  is done")


'''This function computes sequences of episodes '''


def find_sequences(clusterer, clusterer_name, evaluation, location, nb_clusters):
    # driving scenarios
    S1 = "Braking_before_crossing_or_turning"
    S2 = "Accelerating_after_crossing_or_turning"
    S3 = "Car_turning"
    S4 = "Approaching_a_crossroad_or_an_obstacle"
    S5 = "Car_stopping_at_red_traffic_light_or_traffic_jam"
    S6 = "Car_accelerating_after_stopping_at_red_traffic_light_or_traffic_jam"
    S7 = "Driving_straight_on_a_highway"
    S8 = "Car_in_a_roundabout"
    S9 = "Car_braking"
    S10 = "Car_accelerating"
    S11 = "Car_driving_at_a_constant_speed"

    if location == "Gaimersheim":
        if nb_clusters == 4:
            labels_order = [S2, S1, S3, S4]
            eg_abstract_TC = [S4, S1, S3, S2, S4]
        if nb_clusters == 8:
            labels_order = [S4, S8, S3, S11, S2, S1, S10, S9]
            eg_abstract_TC = [S4, S1, S3, S2, S4, S9, S8, S2, S10]
    if location == "Munich":
        if nb_clusters == 4:
            labels_order = [S5, S2, S4, S3]
            eg_abstract_TC = [S5, S2, S4, S3, S2]
        if nb_clusters == 8:
            labels_order = [S5, S4, S4, S2, S6, S9, S3, S1]
            eg_abstract_TC = [S4, S1, S5, S6, S3, S2, S9, S1, S5]
    if location == "Ingolstadt":
        if nb_clusters == 4:
            labels_order = [S3, S5, S4, S2]
            eg_abstract_TC = [S5, S2, S4, S3, S2]
        if nb_clusters == 8:
            labels_order = [S11, S3, S6, S4, S1, S5, S3, S4]
            eg_abstract_TC = [S4, S1, S5, S6, S3, S11, S1, S5]

    clusters_labels = []
    for cluster in range(0, clusterer.number_of_clusters):
        clusters_labels.append([cluster, labels_order[cluster]])

    sequence_actions = []
    cluster_ids = []

    for inst in range(len(evaluation.cluster_assignments) - 1):
        cluster_ids.append(evaluation.cluster_assignments[inst])
        if evaluation.cluster_assignments[inst + 1] != evaluation.cluster_assignments[inst] and len(
                cluster_ids) > 20:

            for label in clusters_labels:
                if cluster_ids[0] == label[0]:
                    action = label[1]
                    sequence_actions.append(action)
                    cluster_ids = []
                    break

    compute_transitions(sequence_actions, labels_order, location, nb_clusters, clusterer_name)

    print("Example of TC concretization")
    # verify that the Driving scenarios in the example of abstract Test case must be seen in the observed scenarios
    # in the location with the selected number of clusters set up the path to the file containing the  example of
    # abstract test cases generated by Graph walker otherwise use the eg_abstractTC variables
    path_to_test_steps_file = "results/" + location + "/GW_testpath_k" + str(nb_clusters)+".txt"
    if os.path.exists(
            path_to_test_steps_file):
        print("using the GW files")
        eg_abstract_TC = get_abstrat_TC(path_to_test_steps_file,labels_order)

    TCs_Concretization(location, eg_abstract_TC, nb_clusters, labels_order, clusterer_name)


def compute_transitions(sequence_actions, labels_order, location, nb_clusters, clusterer_name):
    if not os.path.exists(os.path.join("results/", location, clusterer_name, "Automata_files/")):
        os.makedirs(os.path.join("results/", location, clusterer_name, "Automata_files/"))
    with open('results/' + location + "/" + clusterer_name + "/Automata_files/" +
              'Transitions_dict_' + location + '_K' + str(nb_clusters) + '.json',
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
        create_automata_json_dot_files(sequence_actions, labels_order, transitions, location, nb_clusters,
                                       clusterer_name)


def create_automata_json_dot_files(sequence_actions, labels_order, transitions, location, nb_clusters, clusterer_name):
    transitions_list = []
    alphabet = []
    for key in transitions:
        for s_out in labels_order:
            if transitions[key][s_out]['nbr_edges'] > 0:
                edge_name = "e_" + str(transitions[key][s_out]['nbr_edges'])
                T = [key, edge_name, s_out]
                alphabet.append(edge_name)
                transitions_list.append(T)
    # print("transition list ", transitions_list)
    # print("sequence of actions", sequence_actions)
    with open('results/' + location + "/" + clusterer_name + "/Automata_files/" +
              'Automata_' + location + '_K' + str(nb_clusters) + '.json',
              'w') as f:
        # print("initial node", sequence_actions[0])
        digraph = {"alphabet": alphabet, "states": [], "initial_state": sequence_actions[0], "accepting_states": [],
                   "transitions": []}
        for s in labels_order:
            digraph["states"].append(s)

        digraph["transitions"] = transitions_list

        # print(digraph)
        json.dump(digraph, f, sort_keys=True, indent=4)

    dfa_example = PySimpleAutomata.automata_IO.dfa_json_importer(
        'results/' + location + "/" + clusterer_name + "/Automata_files/" +
        'Automata_' + location + '_K' + str(nb_clusters) + '.json')

    automata_IO.dfa_to_dot(dfa_example, 'automata_graph_' + location + '_K' + str(nb_clusters),
                           'results/' + location + "/" + clusterer_name + "/Automata_files/")

    automata_IO.dfa_to_json(dfa_example, 'automata_graph_' + location + '_K' + str(nb_clusters),
                            'results/' + location + "/" + clusterer_name + "/Automata_files/")


def get_abstrat_TC(path_to_test_steps_file, label_order):
    abstract_TC = []
    with open(path_to_test_steps_file) as f:
        for line in f:
            x = line[:-1]
            for label in label_order:
                if label in x:
                    abstract_TC.append(label)
    # print(abstract_TC)
    return (abstract_TC)


''' Performing Linear Inteprolation in order to concretize abstract test cases'''

def TCs_Concretization(location, abstract_TC, nb_clusters, labels_order, clusterer_name):
    if not os.path.exists(os.path.join("results/", location, clusterer_name, "Concretized_test_cases/")):
        os.makedirs(os.path.join("results/", location, clusterer_name, "Concretized_test_cases/"))

    """ labels_order_Gaimersheim = [S2, S1, S3, S4]
     e.g. abstract_TC = [S4,S1,S3,S2,S4]
    """
    # load input data and cluster assignment
    input_data = get_input_data(location)
    clean_data = get_clean_data(location)
    Cluster_assignments = get_clusters_assignments(clusterer_name, location, nb_clusters)
    conctretized_TC = {}
    for att in input_data.attribute_names():
        conctretized_TC[att] = {'timestamps': [],
                                'values': [],
                                'Labels': [],
                                'Label_idx': []}

    with open('results/' + location + "/" + clusterer_name + "/Concretized_test_cases"
                                                             '/Concretized_TC_' + location + '_K' + str(
        nb_clusters) + '.json',
              'w') as file:

        for att in range(0, input_data.num_attributes):
            att_name = input_data.attribute_names()[att]
            selected_episodes = []
            if abstract_TC[0] in labels_order:
                c = labels_order.index(abstract_TC[0])
                List_EPs_0 = ExtractEpisodes_withTimestamps(input_data, clean_data, c, Cluster_assignments, att)
                selected_episodes.append(List_EPs_0[0])

            for idx_s in range(1, len(abstract_TC)):
                if abstract_TC[idx_s] in labels_order:

                    for label in labels_order:
                        if abstract_TC[idx_s] == label:
                            c = labels_order.index(label)
                            List_EPs_idx = ExtractEpisodes_withTimestamps(input_data, clean_data, c,
                                                                          Cluster_assignments,
                                                                          att)

                    for ep in range(0, len(List_EPs_idx)):
                        # Note: the timestamps of Ep_1 must be smaller than timestamps of Ep_2
                        if checkTS(selected_episodes[len(selected_episodes) - 1][1], List_EPs_idx[ep][1]):
                            selected_episodes.append(List_EPs_idx[ep])
                            break

            time_axis = []
            values_axis = []
            labels = []
            idx_labels = []
            for ep in range(0, len(selected_episodes)):
                for t in selected_episodes[ep][1]:
                    time_axis.append(t)
                    labels.append(abstract_TC[ep])
                    idx_labels.append(int(labels_order.index(abstract_TC[ep])))
                for v in selected_episodes[ep][0]:
                    values_axis.append(v)

            new_timeline = []
            min_diff = time_axis[1] - time_axis[0]
            for i in range(1, len(time_axis)):
                if time_axis[i] - time_axis[i - 1] < min_diff:
                    min_diff = time_axis[i] - time_axis[i - 1]

            new_timeline.append(min(time_axis))
            i = 0
            while new_timeline[i] < max(time_axis):
                new_timeline.append(new_timeline[i] + min_diff)
                i = i + 1

            new_timeline[len(new_timeline) - 1] = max(time_axis)

            x = time_axis
            y = values_axis
            f = interpolate.interp1d(x, y, kind='linear')
            y_new = f(new_timeline)
            # plt.plot(new_timeline, y_new, '.', label='Interpolated parts')

            f2 = interpolate.interp1d(x, idx_labels, kind='previous')
            new_idx_labels = f2(new_timeline)

            new_labels = []
            for idx in new_idx_labels:
                for lab in labels_order:
                    if int(idx) == int(labels_order.index(lab)):
                        new_labels.append(lab)
                        break  # because there might be repetitive cluster labels

            data = {'timestamps': new_timeline,
                    'values': y_new,
                    'Labels': new_labels,
                    'Label_idx': new_idx_labels}

            df = pd.DataFrame(data)
            plt.title(input_data.attribute(att))
            groups = df.groupby(df['Labels'])
            for name, group in groups:
                plt.plot(group['timestamps'], group['values'], marker='o', linestyle='', markersize=3, label=name)
            plt.legend(loc='upper left')
            plt.show()

            conctretized_TC[att_name]['timestamps'].append(list(new_timeline))
            conctretized_TC[att_name]['values'].append(list(y_new))
            conctretized_TC[att_name]['Labels'].append(list(new_labels))
            conctretized_TC[att_name]['Label_idx'].append(list(new_idx_labels))

        json.dump(conctretized_TC, file, sort_keys=True, indent=4)


def checkTS(list1, list2):
    for i in list2:
        return all(x < i for x in list1)


def get_clean_data(location):
    # load data sets
    loader = Loader("weka.core.converters.ArffLoader")
    path_to_clean_data = os.path.join(
        "data/Clustering_input_data/",
        location,
        "test_data_cleaned.arff")
    clean_data = loader.load_file(path_to_clean_data)
    return clean_data


def get_input_data(location):
    # load data sets
    loader = Loader("weka.core.converters.ArffLoader")
    path_to_filtered_data = os.path.join(
        "data/Clustering_input_data/",
        location,
        "test_data_Filtered.arff")
    input_data = loader.load_file(path_to_filtered_data)
    return input_data


def save_cluster_assignments(clusterer_name, evaluation, location, nb_clusters):
    print("Saving clusters assginments for ", location, " if not already saved")
    if not os.path.exists(os.path.join("results/", location, clusterer_name + "/Clusters_assignments/")):
        os.makedirs(os.path.join("results/", location, clusterer_name + "/Clusters_assignments/"))

    cluster_data_path = 'results/' + location + "/" + clusterer_name + "/Clusters_assignments/"

    # save list in file
    with open(
            cluster_data_path + location + '_Cluster_assignments_K' + str(
                nb_clusters) + '.txt',
            'w') as f:
        for val in evaluation.cluster_assignments:
            f.write("%s\n" % str(val))

    # load list from file
    Cluster_assignments_list = []
    if os.path.exists(
            cluster_data_path + location + '_Cluster_assignments_K' + str(
                nb_clusters) + '.txt'):
        with open(cluster_data_path + location + '_Cluster_assignments_K' + str(
                nb_clusters) + '.txt') as f:
            for line in f:
                x = line[:-1]

                # add current item to the list
                Cluster_assignments_list.append(int(float(x)))


def get_clusters_assignments(clusterer_name, location, nb_clusters):
    cluster_data_path = 'results/' + location + "/" + clusterer_name + "/Clusters_assignments/"
    Cluster_assignments = []
    if os.path.exists(
            cluster_data_path + location + '_Cluster_assignments_K' + str(
                nb_clusters) + '.txt'):
        with open(
                cluster_data_path + location + '_Cluster_assignments_K' + str(
                    nb_clusters) + '.txt') as f:
            for line in f:
                x = line[:-1]
                Cluster_assignments.append(int(float(x)))

    return Cluster_assignments


def ExtractEpisodes_withTimestamps(data_filtered, clean_data, cluster, evaluation, att):
    List_episodes = []
    val_list = []
    TS_list = []
    for inst in range(len(evaluation) - 1):
        if evaluation[inst] == cluster:
            val_list.append(data_filtered.get_instance(inst).get_value(att))
            TS_list.append(clean_data.get_instance(inst).get_value(0))
            if evaluation[inst + 1] != cluster and len(val_list) > 200:
                List_episodes.append([val_list, TS_list])
                val_list = []
                TS_list = []

    # print(len(List_episodes[0][0]),len(List_episodes[0][1]))
    return List_episodes


"""
 Please Change the right location name,
 the available locations are "Gaimersheim", "Munich" and "Ingolstadt"

"""

if __name__ == "__main__":
    try:
        jvm.start(system_cp=True, packages=True, max_heap_size="512m")
        datasets = ["Gaimersheim", "Ingolstadt", "Munich"]
        for location in datasets:
            main(location)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
