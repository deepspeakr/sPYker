import configparser
import os

from utility.data_util import DatasetControl, Speaker

# from utility.train_test_models import train_model

config = configparser.ConfigParser()
config.read("./config.ini")
config.sections()

training_source = config["DIRS"]["training_source"]
training_dest = config["DIRS"]["training_dest"]
testing_source = config["DIRS"]["testing_source"]

dataset = ".\\train-clean-100\\"

DatasetControl.clear_dest_dirs()

DatasetControl.extract_from_subdirs_librispeech(dataset)

DatasetControl.move_train_test_data(dataset, testing_source, training_source)

# DatasetControl.add_noise_method(training_source, -20)

# DatasetControl.normalize(training_source)

Speaker.train_test_ratio(dataset, 10)

# train_model()

# from csv_wrapper import generate_csv

# generate_csv()
