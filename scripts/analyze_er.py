#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import os
import collections
import pandas
import sklearn.metrics

def score(patients, ef, pred):
    a = []
    b = []

    for patient in patients:
        a.append(ef[patient])
        b.append(pred[patient])

    return sklearn.metrics.roc_auc_score(a, b), 0

def bootstrap(ef, pred, n=1000):
    point = score(pred.keys(), ef, pred)

    patients = list(pred.keys())
    samples = []
    for _ in range(n):
        samples.append(score([random.choice(patients) for _ in range(len(patients))], ef, pred))

    samples = list(map(sorted, zip(*samples)))
    return [(p, s[round(0.025 * len(s))], s[round(0.975 * len(s))]) for (p, s) in zip(point, samples)]


def main(src="output/er/ef"):
    ef = collections.defaultdict(dict)
    interpretable = collections.defaultdict(dict)
    # with open("data/er/split_0/FileList.csv") as f:
    with open("data/er/FileList.csv") as f:
        assert f.readline() == "FileName,EF,Interpretable,Split\n"
        for line in f:
            filename, e, i = line.strip().split(",")
            ef[filename] = e
            interpretable[filename] = i

    for method in [
        "lr_1e-4",
        "blind",
        "scratch",
        # "lr_1e-5",
        # "lr_1e-6",
        "transfer",
    ]:

        print(method)
        print("=" * len(method))
        seeds = list(range(10))
        pred = {}
        for seed in seeds:
            p = load(os.path.join(src, "{}_{}".format(method, seed), "test_predictions.csv"))
            assert p.keys() == (p.keys() - pred.keys())
            pred.update(p)

        # pred = {patient: sum(pred[view][patient][accession]) / len(pred[view][patient][accession]) for accession in pred[view][patient] if pred[view][patient][accession] != []} for patient in pred[view]}
        pred = {patient: np.array(pred[patient]).mean(0) for patient in pred}  # TODO: meaning in logit space is a bit weird
        ef_hat = {patient: pred[patient][0] for patient in pred}
        interpretable_hat = {patient: pred[patient][1] for patient in pred}

        print("EF")
        for (score, (p, l, h)) in zip(["AUC", "CE"], bootstrap({patient: 1 if ef[patient] == "Normal" else 0 for patient in ef if interpretable[patient] != "No"}, {patient: ef_hat[patient] for patient in ef_hat if interpretable[patient] != "No"})):
            print("{}: {:.2f} ({:.2f} - {:.2f})".format(score, p, l, h))
        print()

        fig = plt.figure(figsize=(3, 3))
        fpr, tpr, _ = sklearn.metrics.roc_curve([ef[patient] == "Normal" for patient in sorted(ef_hat) if interpretable[patient] != "No"], [ef_hat[patient] for patient in sorted(ef_hat) if interpretable[patient] != "No"])
        plt.plot(fpr, tpr, color="black")
        plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("EF ROC")
        plt.tight_layout()
        des = "output/er_analyze/fig"
        os.makedirs(des, exist_ok=True)
        plt.savefig(os.path.join(des, "ef_{}.pdf".format(method)))


        print("Severely Reduced")
        count = collections.Counter()
        for patient in ef_hat:
            if interpretable[patient] != "No":
                count[ef[patient], ef_hat[patient] > 0] += 1
                if ef[patient] == "Severely Reduced" and ef_hat[patient] > 0:
                    print(os.path.splitext(patient)[0] + ".*", end=" ")
        print()

        print("Moderately Reduced")
        count = collections.Counter()
        for patient in ef_hat:
            if interpretable[patient] != "No":
                count[ef[patient], ef_hat[patient] > 0] += 1
                if ef[patient] == "Moderately Reduced" and ef_hat[patient] > 0:
                    print(os.path.splitext(patient)[0] + ".*", end=" ")
        print()

        print("Normal")
        count = collections.Counter()
        for patient in ef_hat:
            if interpretable[patient] != "No":
                count[ef[patient], ef_hat[patient] > 0] += 1
                if ef[patient] == "Normal" and ef_hat[patient] < 0:
                    print(os.path.splitext(patient)[0] + ".*", end=" ")
        print()

        for score in ["Normal", 'Slightly Reduced', 'Moderately Reduced', 'Severely Reduced']:
            print(score, count[score, True], count[score, False])


        print("Interpretable")
        for (score, (p, l, h)) in zip(["AUC", "CE"], bootstrap({patient: 1 if interpretable[patient] != "No" else 0 for patient in interpretable}, interpretable_hat)):
            print("{}: {:.2f} ({:.2f} - {:.2f})".format(score, p, l, h))
        print()

    #     merged = collections.defaultdict(lambda : {})
    #     for patient in pred[views[0]]:
    #         for accession in pred[views[0]][patient]:
    #             p = []
    #             for view in views:
    #                 try:
    #                     p.append(pred[view][patient][accession])
    #                 except KeyError:
    #                     break
    #             if len(p) == len(views):
    #                 merged[patient][accession] = sum(p) / len(p)
    #     print("Both Views")
    #     for (score, (p, l, h)) in zip(["R2", "RMSE"], bootstrap(ef, merged)):
    #         print("{}: {:.2f} ({:.2f} - {:.2f})".format(score, p, l, h))
    #     print()


def load(filename):
    if not os.path.isfile(filename):
        print(filename, "not found.")
        return {}
    pred = collections.defaultdict(list)
    with open(filename) as f:
        for line in f:
            patient, frame, *p = line.split(",")
            pred[patient].append(tuple(map(float, p)))
    return pred

if __name__ == "__main__":
    main()
