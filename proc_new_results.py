import configparser
import os

from utility.data_util import DatasetControl, Speaker
from utility.train_test_models import train_model

config = configparser.ConfigParser()
config.read("./config.ini")
config.sections()

training_source = config["DIRS"]["training_source"]
training_dest = config["DIRS"]["training_dest"]
testing_source = config["DIRS"]["testing_source"]

dataset = ".\\train-clean-100\\"

os.makedirs(training_source) if not os.path.exists(training_source) else print(
    f"{training_source} istnieje"
)
os.makedirs(training_dest) if not os.path.exists(training_dest) else print(
    f"{training_dest} istnieje"
)
os.makedirs(testing_source) if not os.path.exists(testing_source) else print(
    f"{testing_source} istnieje"
)


DatasetControl.clear_dest_dirs()

# DatasetControl.extract_from_subdirs_voxceleb(dataset)
# DatasetControl.extract_from_subdirs_timit(dataset)
DatasetControl.extract_from_subdirs_librispeech(dataset)

DatasetControl.move_train_test_data(dataset, testing_source, training_source)

# DatasetControl.add_noise_method(training_source, -20)

# DatasetControl.normalize(training_source)

Speaker.train_test_ratio(dataset, 5)

train_model()


from csv_wrapper import generate_csv

generate_csv()
