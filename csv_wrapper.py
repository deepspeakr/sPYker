import csv

from utility.train_test_models import test_model, get_models

test_samples, winners = test_model()

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


def generate_csv():
    for model in get_models():
        models.append(Result(model.split(".gmm")[0], 0, 0))

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

    with open("wyniki_identyfikacji.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "Speaker ID",
                "liczba poprawnych detekcji",
                "liczba testowanych plikow",
                "udzial procentowy [%]",
            ]
        )
        for model in models:
            writer.writerow(
                [
                    model.id,
                    model.passed,
                    model.all,
                    (model.passed / model.all) * 100,
                ]
            )

    print("\n\nWyniki wygenerowane do pliku wyniki_identyfikacji.csv\n\n")
