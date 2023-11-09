# Clustering_Vehicle_data

This repository includes modified data and source code of scripts we used for obtaining the experimental results in the following publications: 

1. Chetouane, Nour, Lorenz Klampfl, and Franz Wotawa. "Extracting information from driving data using k-means clustering." Proceedings of the 33rd International Conference on Software Engineering & Knowledge Engineering (SEKE), KSIR Virtual Conference Center, Pittsburgh, USA. 2021.

2. Chetouane, Nour, and Franz Wotawa. "On the application of clustering for extracting driving scenarios from vehicle data." Machine Learning with Applications 9 (2022): 100377.

3. Chetouane, Nour, and Franz Wotawa. "Extracting temporal models from data episodes." 2022 IEEE 22nd International Conference on Software Quality, Reliability and Security (QRS). IEEE, 2022.

4. Chetouane, Nour, and Franz Wotawa. "Generating concrete test cases from vehicle data using models obtained from clustering." 2023 IEEE International Conference on Software Testing, Verification and Validation Workshops (ICSTW). IEEE, 2023.

5. Chetouane, Nour, and Franz Wotawa. "Using Data Abstraction for Clustering in the Context of Test Case Generation." Accepted and to be published in  2023 IEEE 22nd International Conference on Software Quality, Reliability and Security (QRS). IEEE, 2023.
   
# Original data:

In this work, we make use of the public available A2D2 data [1]  originally downloaded from https://www.a2d2.audi/a2d2/en/download.html

For the clustering approach, we used the bus signal data available for three German cities: Gaimersheim, Munich, and Ingolstadt.

For the driving scenarios validation, we make use of the camera front-center images, which we also downloaded from the A2D2 data link given above.

This repository includes the original bus signal .json files for the three German cities. Due to limited space of memory, we only provide the camera image files in .json format for the Gaimersheim dataset. To download camera image files in .png format and the .json for the rest of the German cities,  we refer to the above-mentioned link to the A2D2 data.

For our study, we performed changes on the original bus signal data. We provide the modified data used in each step of our approach in the following folders; Interpolated data, arff data, and clustering_input data.

[1] Geyer, Jakob, et al. "A2d2: Audi autonomous driving dataset." arXiv preprint arXiv:2004.06320 (2020).

# Necessary Python Libraries:

To be able to run all algorithms available in the Python wrapper for the Java machine learning workbench Weka (see: https://www.cs.waikato.ac.nz/~ml/weka/), it is mandatory to install the following libraries:

1. python-weka-wrapper3: https://github.com/fracpete/python-weka-wrapper3

2. python-javabridge 4.0.3: https://pypi.org/project/python-javabridge/

3. sklearn-weka-plugin:  This plugin makes Weka algorithms available in scikit-learn. https://github.com/fracpete/sklearn-weka-plugin

4. PySimpleAutomata: We used this library to create cluster-episode graphs. https://pysimpleautomata.readthedocs.io/en/latest/index.html



# Repository structure

This framework for carrying out our experiments comprises the following three parts:

I. Data preprocessing:

For data preprocessing, we provide the following scripts:

1. Data_interpolation.py: This script performs Cubic spline interpolation on the original data to synchronize all bus signal values. The interpolated data for each city is saved in the  "Interpolated_data" folder.

2. Convert_Interpolated_data.py: This script converts the interpolated data into the .arff format which is the required format used in Weka.
   In this work, we only make use of four main signals which are: 'accelerator_pedal', 'brake_pressure', 'steering_angle_calculated', and 'vehicle_speed'. It is possible to use all signals provided in the A2D2 data set if required. The data in .arff format for each city is saved in the "arff_data" folder.

3. Data_cleansing.py: This script replaces all brake pressure values which are below or equal to 0.2  with the value 0 to obtain more precise results. It also filters out the timestamps attribute values, which are not considered when clustering the data. The clean data (including timestamps) and filtered data (without timestamps) are saved in the "Clustering-input_data" folder"

II. Data clustering:

For data clustering we provide two scripts:

1. Data_clustering: In this script, we perform clustering using five different algorithms provided by Weka (Simple k-means, EM, Canopy clustering, SOM clustering, and K-means using Autoencoder pre-processing ). We save the extracted driving scenarios in .json files using the function: create_JSON_files, and we provide the option of saving driving scenarios in .arff format if needed to be used as input to Weka. For this, we use the function Create_arff_outputfiles. The extracted driving scenarios are saved in both formats in the results/<location>/driving_scenarios folder. <br>  In this script, we further compute the average percentage of Pearson correlation for each cluster using the function AVG_Percentage_Pearson_correlation. The Pearson correlation results for each city are saved in .txt format in the results/location_name/Pearson_correlation_results folder. We also provide some other graphs of extracted episodes, their probability distribution, and Pearson correlation matrices for each attribute obtained in a previous study (see: https://web.archive.org/web/20211013034723id_/http://ksiresearch.org/seke/seke21paper/paper118.pdf). These plots can be saved in results/location_name/plots folder. <br>
<strong>Extracting automata graphs:</strong> In addition, we create automata graphs using PySimpleAutomata library. The functions for creating the graphs is called find_sequences. It computes the sequence of all extracted episodes and calls the functions compute_transitions that computes all transtions between clusters and the function create_automata_json_dot_files which creates the automata models. The .json and .dot files of the automata models are saved in  the directory  results/Location_name/<Clusterer_name>/Automata_files. Note: for now we provide graphs only using k=4 or k=8. <br>
<strong>Test case Generation:</strong> In order to use the automata models for Test cases generation using GraphWalker command line tool (see https://graphwalker.github.io/ ),  it is required to first convert the .dot files into GraphML format. This can be simply done by reading the .dot file and saving it again in .graphml format using yED live (see https://www.yworks.com/yed-live/). It might be as well necessary to further read the .graphml file and save it again using  yEd Graph Editor (see https://www.yworks.com/products/yed/download).
It is important to also change the name of the starting node called automatically by PysimplAutomata library 'fake' to 'start' in order for GraphWalker to recongnize the 'start' node.
   <br> Example of a command used to generate  test cases offline with GraphWlalker CLT: <br>
   <strong> ~username$ java -jar  /mydirectory/graphwalker-cli-4.3.2.jar  offline -m mydirectiry/automata_graph_Gaimersheim_K4.graphml     "quick_random(edge_coverage(100))" </strong> <br>
   In this script, we further use the function called TCs_Concretization, to concretize abstract test cases generated by the graphWlaker CLT. For this, we perform linear interpolation using a common time axis between each selected driving episode corresponding to each cluster/driving scenario.<br>
2. Compute_validity_metrics:  This script computes the Davies-Bouldin index (DB), the Calinski-Harabasz index (CH), and the Silhouette index (S) for each algorithm using different numbers of clusters. These three metrics are available in the Scikit-learn library.

III. Camera images validation:

In this phase, we match camera images available in the original A2D2 data to the extracted driving scenarios in obtained clusters.
Due to the required huge memory space, we only provide the created video for the Gaimersheim example data. The right location needs to be set up inside each of the following 2 scripts:

1. img2vid.py: This script creates videos for each city using the original camera front-center images we downloaded from the A2D2 data link given above. Due to the huge size of the camerga images and created vides and because of limited space, we could not provide the created videos of each data set. In order to create the corresponding videos for Gaimersheim, Munich and Ingolstadt examples, you need to download the camera images .png files  from https://www.a2d2.audi/a2d2/en/download.html , also change 
   the location name, date_of_recording, time_of_recording and type of camera for each example. 

2. synchro_vid_clustered_bus: This script synchronizes the clustered bus signals with camera images and matches each driving scenario (available in results/<location>/driving_scenarios) to corresponding sequences of images. Before running the image_validation() function  in the main.py script, please change the right parameters in synchro_vid_clustered_bus.py script which are the location name,  vid_path, cluster_scenario_path  and cam_data_path. 

Note that it is possible to either run the main.py script once to perform all pre-processing and clustering steps, and the camera image validation, or to run each script separately. For the latter, you need to set up the right location name in each script. 
  
