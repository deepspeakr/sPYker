import os, shutil, random, math
from pydub import AudioSegment, effects
from scipy.io.wavfile import read, write
import numpy as np

from csv_wrapper import generate_csv


import matplotlib.pyplot as plt
from utility.train_test_models import train_model

destination = ".\\training_set\\"
dataset = ".\\VoxCeleb1\\"

train_dest = ".\\training_set\\"
test_dest = ".\\testing_set\\"
trained_models = ".\\trained_models\\"

success_rate_all = []

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = ROOT_DIR.split("\\")[:-1]
ROOT_DIR = "\\".join(ROOT_DIR)
print(ROOT_DIR)

samples_per_speaker = 11


class DatasetControl:
    def extract_from_subdirs_voxceleb(dataset):
        for speaker in os.listdir(dataset):
            sample_count = 1
            if speaker == ".DS_Store":
                continue
            for sub_folder in os.listdir(dataset + speaker + "\\"):
                if os.path.isdir(dataset + speaker + "\\" + sub_folder + "\\"):
                    for speaker_sample in os.listdir(
                        dataset + speaker + "\\" + sub_folder + "\\"
                    ):
                        source = (
                            ROOT_DIR
                            + dataset[1:]
                            + speaker
                            + "\\"
                            + sub_folder
                            + "\\"
                            + speaker_sample
                        )
                        shutil.copy(
                            source,
                            dataset + speaker + "\\sample" + str(sample_count) + ".wav",
                        )

                        print(
                            dataset
                            + speaker
                            + "\\sample"
                            + str(sample_count)
                            + ".wav przetworzono!",
                        )
                        sample_count += 1

    @staticmethod
    def extract_from_subdirs_librispeech(dataset):
        for speaker in os.listdir(dataset):
            sample_count = 1
            if speaker == ".DS_Store":
                continue
            for sub_folder in os.listdir(dataset + speaker + "\\"):
                if os.path.isdir(dataset + speaker + "\\" + sub_folder + "\\"):
                    for speaker_sample in os.listdir(
                        dataset + speaker + "\\" + sub_folder + "\\"
                    ):
                        if speaker_sample.endswith(".flac"):
                            source = (
                                ROOT_DIR
                                + dataset[1:]
                                + speaker
                                + "\\"
                                + sub_folder
                                + "\\"
                                + speaker_sample
                            )
                            file_path = (
                                dataset
                                + speaker
                                + "\\"
                                + sub_folder
                                + "\\"
                                + speaker_sample
                            )
                            flac_tmp_audio_data = AudioSegment.from_file(
                                file_path, format="flac"
                            )
                            flac_tmp_audio_data.export(
                                dataset
                                + speaker
                                + "\\sample"
                                + str(sample_count)
                                + ".wav",
                                format="wav",
                            )
                            print(
                                dataset
                                + speaker
                                + "\\sample"
                                + str(sample_count)
                                + ".wav exported!",
                            )
                            # shutil.copy(
                            #     source,
                            #     dataset + speaker + "\\sample" + str(sample_count) + ".flac",
                            # )
                            sample_count += 1

    def copy_dataset():
        for speaker in os.listdir(dataset):
            if speaker == ".DS_Store":
                continue
            for sub_folder in os.listdir(dataset + speaker + "\\"):
                for speaker_sample in os.listdir(
                    dataset + speaker + "\\" + sub_folder + "\\"
                ):
                    source = (
                        ROOT_DIR
                        + dataset
                        + speaker
                        + "\\"
                        + sub_folder
                        + "\\"
                        + speaker_sample
                    )
                    # shutil.copy(
                    #     source, destination + speaker + "-sample" + str(sample_count) + ".wav"
                    # )
                    # if sample_count >= samples_per_speaker:
                    #     sample_count = 1
                    #     break

    @staticmethod
    def move_train_test_data(dataset, test_destination, train_destination):
        with open("results.txt", "a") as file:
            file.write("Ilosc mowcow \t Skutecznosc \n")

        speaker_cnt = 1
        for speaker in os.listdir(dataset):
            if speaker == ".DS_Store":
                continue
            samples = []

            # moving test files
            for sample in os.listdir(dataset + speaker + "\\"):
                if sample == ".DS_Store":
                    continue
                if not os.path.isdir(dataset + speaker + "\\" + sample + "\\"):
                    samples.append(sample)
            testing_set_list = random.sample(
                samples, Speaker.get_test_count(dataset, speaker, 10)
            )
            test_index = 1
            for test_item in testing_set_list:
                shutil.move(
                    dataset + speaker + "\\" + test_item,
                    test_destination + speaker + "-sample" + str(test_index) + ".wav",
                )
                samples.remove(test_item)
                test_index += 1

            # moving train files
            train_index = 1
            for train_item in samples:
                shutil.move(
                    dataset + speaker + "\\" + train_item,
                    train_destination + speaker + "-sample" + str(train_index) + ".wav",
                )
                train_index += 1

            train_model()

            success_rate = generate_csv(speaker, str(speaker_cnt))

            for item in os.listdir(train_dest):
                os.remove(train_dest + item)

            with open("results.txt", "a") as file:
                file.write(f"{speaker_cnt} \t {success_rate * 100}\n")

            success_rate_all.append(success_rate)

            speaker_cnt += 1

        print(success_rate_all)
        plt.plot(success_rate_all)
        plt.show()

    def clear_dest_dirs():
        for item in os.listdir(train_dest):
            os.remove(train_dest + item)

        for item in os.listdir(trained_models):
            os.remove(trained_models + item)

        for item in os.listdir(test_dest):
            os.remove(test_dest + item)

        os.remove("results.txt")

    @staticmethod
    def add_noise_SNR_destination(dest, snr):
        for sample in os.listdir(dest):
            if sample.endswith(".wav"):
                filename = dest + sample
                sr, signal = read(filename)
                average_power = np.mean(signal**2)
                averagepower_db = 10 * np.log10(average_power)
                noise_db = averagepower_db - snr
                noise_watt = 10 ** (noise_db / 10)
                noise = np.random.normal(0, np.sqrt(noise_watt), signal.shape[0])

                noise_signal = signal + noise

                write(filename, sr, noise_signal)
                print(f" Noise added to {filename}")

    @staticmethod
    def add_noise_module(clean_file, snr):
        for sample in os.listdir(clean_file):
            if sample.endswith(".wav"):
                sr, signal = read(clean_file + sample)
                noise = np.random.normal(0, 0.1, signal.shape[0])
                write(".\\noise.wav", sr, noise.astype(np.int16))
                Noise.add_noise(clean_file + sample, ".\\noise.wav", snr)
                print(f" Noise added to {clean_file + sample}")

    @staticmethod
    def add_noise_for_waveform(s, n, db):
        alpha = np.sqrt(np.sum(s**2) / (np.sum(n**2) * 10 ** (db / 10)))
        mix = s + alpha * n
        return mix

    @staticmethod
    def add_noise_method(file, snr):
        for sample in os.listdir(file):
            if sample.endswith(".wav"):
                sr, signal = read(file + sample)
                noise = np.random.normal(0, 0.1, signal.shape[0])
                mixed = DatasetControl.add_noise_for_waveform(signal, noise, snr)
                write(file + sample, sr, mixed.astype("int16"))
                print(f" Noise added to {file + sample}")

    @staticmethod
    def match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    @staticmethod
    def normalize(file):
        for sample in os.listdir(file):
            if sample.endswith(".wav"):
                rawsound = AudioSegment.from_file(file + sample, "wav")
                normalized_sound = DatasetControl.match_target_amplitude(
                    rawsound, -20.0
                )
                normalized_sound.export(file + sample, format="wav")
                print(f"{file + sample} normalized!")


class Speaker:
    def __init__(self, id, sample_count) -> None:
        self.id = id
        self.s_count = sample_count

    def count_speaker_samples(dataset):
        speakers = []
        for speaker in os.listdir(dataset):
            sample_count = 0
            if speaker == ".DS_Store":
                continue
            for sub_folder in os.listdir(dataset + speaker + "\\"):
                if os.path.isdir(dataset + speaker + "\\" + sub_folder + "\\"):
                    for _ in os.listdir(dataset + speaker + "\\" + sub_folder + "\\"):
                        sample_count = sample_count + 1

            speakers.append(Speaker(speaker, sample_count))
        return speakers

    @staticmethod
    def train_test_ratio(dataset, n):
        speakers = Speaker.count_speaker_samples(dataset)
        for x in range(len(speakers)):
            testing_set_count = int(speakers[x].s_count / (n + 1))
            print(
                f"{speakers[x].id}: sample count: {speakers[x].s_count}\t->\ttesting count: {testing_set_count}"
            )
        return testing_set_count

    def get_speaker_count():
        count = 0
        for speaker in os.listdir(dataset):
            if speaker != ".DS_Store":
                count += 1
        return count

    @staticmethod
    def get_test_count(dataset, speaker, n):
        speakers = Speaker.count_speaker_samples(dataset)
        for speak in speakers:
            if speak.id == speaker:
                return int(speak.s_count / (n + 1))


# if __name__ == "__main__":
#     # DatasetControl.extract_from_subdirs()
#     librispeech = ".\\train-clean-100\\"
#     # DatasetControl.extract_from_subdirs_librispeech(librispeech)

#     # DatasetControl.add_noise_SNR_destination(train_dest, 20)

#     DatasetControl.add_noise_method(train_dest, -20)

#     DatasetControl.normalize(train_dest)

#     # DatasetControl.clear_dest_dirs()

#     Speaker.train_test_ratio(librispeech, 10)

#     # DatasetControl.move_train_test_data(librispeech, test_dest, train_dest)


if __name__ == "__main__":
    from add_noise import Noise

    DatasetControl.clear_dest_dirs()
else:
    from utility.add_noise import Noise
