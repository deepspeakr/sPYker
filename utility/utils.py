import os


class DatasetManipulation:
    def extract_name_count(dir: str) -> dict:
        speaker_list = []
        sample_count = []
        for sample in os.listdir(dir):
            speaker_id = sample.split("-")[0]
            if speaker_id not in speaker_list:
                speaker_list.append(speaker_id)
        for speaker in speaker_list:
            count = 1
            for speaker_sample in os.listdir(dir):
                if speaker_sample.split("-")[0] == speaker:
                    count = count + 1
            sample_count.append(count - 1)
        output = {}
        output.fromkeys(speaker_list)
        index = 0
        for speaker in speaker_list:
            output[speaker] = sample_count[index]
            index = index + 1

        return output
