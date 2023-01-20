import traceback
import weka.core.jvm as jvm
import Compute_validity_metrics
import Convert_Interpolated_Data
import Data_Clustering
import Data_cleansing
import Data_interpolation
import synchro_vid_clustered_bus


# import img2vid


def clustering(Location):
    print(" ***Data pre_processing***")
    print(" 1. Data interpolation")
    # Data_interpolation.main(Location)

    print("2. Convert interpolated data into .ARFF format")
    # Convert_Interpolated_Data.main(Location)

    print("3. Data Cleansing")
    # Data_cleansing.main(Location)

    print("***Data Clustering****")
    print("Computing Pearson coefficient and creating json files for each cluster")
    # Data_Clustering.main(Location)

    print("Measuring validity metrics;  DB index, CH index and Silhouette for each algorithm with different values of "
          "K")
    # Compute_validity_metrics.main(Location)


def images_validation(Location):
    """
    For downloading the original Front center camera images for each city  please refer to
    https://www.a2d2.audi/a2d2/en/download.html .json and .jpeg files are needed in this phase. Due to limited space in
    memory we only provide images of .json format of the Gaimersheim example besides we already created the created
    video for this example.
    """

    print("Creating video using Front camera images ")
    # uncomment  to create other videos for Munich and Ingolstadt examples.
    # Please first download the camera images .json and .png files from https://www.a2d2.audi/a2d2/en/download.html
    # img2vid.main(Location)
    print("Validation using Camera images")


if __name__ == "__main__":

    """
    The original bus signals data of each city is downloaded  from a2d2: https://www.a2d2.audi/a2d2/en/download.html
    the available locations are "Gaimersheim", "Munich" and "Ingolstadt"

    """
    Location = "Gaimersheim"

    try:
        jvm.start(system_cp=True, packages=True, max_heap_size="512m")
        clustering(Location)

    except Exception as e:
        print(traceback.format_exc())

    finally:
        jvm.stop()
        exit()

    # Uncomment  if you want to perform image validation step
    images_validation(Location)
