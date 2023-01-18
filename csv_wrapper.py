import csv

from utility.train_test_models import test_model


models = []
result_structure = []


class Result:
    def __init__(self, id, passed, all) -> None:
        self.id = id
        self.passed = passed
        self.all = all

    def successful_pass(self):
        self.passed += 1
        self.all += 1

    def failed(self):
        self.all += 1

    def reset_object(self):
        self.passed = 0
        self.all = 0


def generate_csv(
    model,
    filename="wyniki_identyfikacji",
):

    test_samples, winners = test_model()
    models.append(Result(model, 0, 0))

    for x in range(len(test_samples)):
        test_samples[x] = test_samples[x].split("-")[0]

    for x in range(len(winners)):
        winners[x] = winners[x].split("/")[-1]

    for x in range(len(test_samples)):
        if test_samples[x] == winners[x]:
            for model in models:
                if model.id == test_samples[x]:
                    Result.successful_pass(model)
        else:
            for model in models:
                if model.id == test_samples[x]:
                    Result.failed(model)

    # for model in models:
    #     print(f"{model.id}: {model.passed}/{model.all}")

    with open(filename + ".csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "Speaker ID",
                "liczba poprawnych detekcji",
                "liczba testowanych plikow",
                "udzial procentowy",
            ]
        )
        for model in models:
            writer.writerow(
                [
                    model.id,
                    model.passed,
                    model.all,
                    str((model.passed / model.all) * 100) + "%",
                ]
            )

        passed_cnt = 0
        all_cnt = 0
        for model in models:
            passed_cnt = passed_cnt + model.passed
            all_cnt = all_cnt + model.all

        writer.writerow(["", "", "", ""])
        writer.writerow(["", "Skutecznosc", (passed_cnt / all_cnt) * 100, ""])

    print("\n\nWyniki wygenerowane do pliku wyniki_identyfikacji.csv\n\n")

    for model in models:
        Result.reset_object(model)

    return passed_cnt / all_cnt


if __name__ == "__main__":
    print("Biblioteka nie runować")
