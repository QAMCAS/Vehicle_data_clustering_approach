import traceback
import arff
from weka.core import jvm
from weka.core.dataset import Instances, Attribute, Instance
import logging
from weka.core.converters import Loader
from weka.filters import Filter, MultiFilter

logger = logging.getLogger(__name__)


def main(location):
    loader = Loader("weka.core.converters.ArffLoader")
    initial_data = loader.load_file('data/arff_data/' + location + '_Selected_att_Clustering_Weka_Inputdata.arff')

    # changing the brake pressure values which are < 0.2 to 0
    print(initial_data.attribute(2))

    for ins in range(0, initial_data.num_instances):
        if initial_data.get_instance(ins).get_value(2) <= 0.2:
            initial_data.get_instance(ins).set_value(2, 0)

    data_cleaned = Instances.copy_instances(initial_data, 0, initial_data.num_instances)

    arff.dump('data/Clustering_input_data/' + location + '/test_data_cleaned.arff',
              data_cleaned,
              relation="Audi",
              names=['timestamps', 'accelerator_pedal', 'brake_pressure', 'steering_angle_calculated', 'vehicle_speed'])

    # Filtering out  attribute 'Timestamps' before  clustering
    filter_TS = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
    multi = MultiFilter()
    multi.filters = [filter_TS]
    multi.inputformat(data_cleaned)
    filtered_multi = multi.filter(data_cleaned)

    data = Instances.copy_instances(filtered_multi, 0, filtered_multi.num_instances)
    arff.dump(
        'data/Clustering_input_data/' + location + '/test_data_Filtered.arff',
        data, relation="Audi",
        names=['accelerator_pedal', 'brake_pressure', 'steering_angle_calculated', 'vehicle_speed'])

    print("done pre-processing the data ")


"""
 Please Change the right location name to run this script separately ,
 the available locations are "Gaimersheim", "Munich" and "Ingolstadt"

"""

if __name__ == "__main__":
    try:
        jvm.start(system_cp=True, packages=True, max_heap_size="512m")
        location = "Gaimersheim"
        main(location)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
        
