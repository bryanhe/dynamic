#!/usr/bin/env python3

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
        for accession in pred[patient]:
            a.append(ef[patient][accession])
            b.append(pred[patient][accession])

    return sklearn.metrics.r2_score(a, b), math.sqrt(sklearn.metrics.mean_squared_error(a, b))

def bootstrap(ef, pred, n=1000):
    point = score(pred.keys(), ef, pred)

    patients = list(pred.keys())
    samples = []
    for _ in range(n):
        samples.append(score([random.choice(patients) for _ in range(len(patients))], ef, pred))

    samples = list(map(sorted, zip(*samples)))
    return [(p, s[round(0.025 * len(s))], s[round(0.975 * len(s))]) for (p, s) in zip(point, samples)]


def main(src="output/ef"):
    ef = collections.defaultdict(dict)
    for view in ["A4C", "PSAX"]:
        with open("data/pediatric/{}/FileList.csv".format(view)) as f:
            assert f.readline() == "FileName,EF,Split\n"
            for line in f:
                filename, e, _ = line.split(",")
                patient, accession, _ = os.path.splitext(filename)[0].split("-")
                e = float(e)

                if accession in ef[patient]:
                    assert ef[patient][accession] == e
                ef[patient][accession] = e

    for method in [
        "blind",
        "scratch",
        "lr_1e-4",
        "lr_1e-5",
        "lr_1e-6",
        "transfer",
        ]:

        print(method)
        print("=" * len(method))
        views = ["A4C", "PSAX"]
        seeds = list(range(10))
        pred = {view: {} for view in views}  # pred[view][patient][accession]
        for view in views:
            for seed in seeds:
                p = load(os.path.join(src, "{}_{}_{}".format(view, seed, method), "test_predictions.csv"))
                assert p.keys() == (p.keys() - pred[view].keys())
                pred[view].update(p)

        pred = {view: {patient: {accession: sum(pred[view][patient][accession]) / len(pred[view][patient][accession]) for accession in pred[view][patient] if pred[view][patient][accession] != []} for patient in pred[view]} for view in pred}
        for view in pred:
            # for patient in pred[view]:
            #     for accession in pred[view][patient]:
            #         pred[view][patient][accession] = sum(pred[view][patient][accession]) / len(pred[view][patient][accession])

            # print(view, score(pred[view].keys(), ef, pred[view]))
            print(view)
            for (score, (p, l, h)) in zip(["R2", "RMSE"], bootstrap(ef, pred[view])):
                print("{}: {:.2f} ({:.2f} - {:.2f})".format(score, p, l, h))
            print()

        merged = collections.defaultdict(lambda : {})
        for patient in pred[views[0]]:
            for accession in pred[views[0]][patient]:
                p = []
                for view in views:
                    try:
                        p.append(pred[view][patient][accession])
                    except KeyError:
                        break
                if len(p) == len(views):
                    merged[patient][accession] = sum(p) / len(p)
        print("Both Views")
        for (score, (p, l, h)) in zip(["R2", "RMSE"], bootstrap(ef, merged)):
            print("{}: {:.2f} ({:.2f} - {:.2f})".format(score, p, l, h))
        print()


def load(filename):
    if not os.path.isfile(filename):
        print(filename, "not found.")
        return {}
    pred = collections.defaultdict(lambda : collections.defaultdict(list))
    with open(filename) as f:
        for line in f:
            filename, frame, p = line.split(",")
            patient, accession, instance = os.path.splitext(filename)[0].split("-")
            pred[patient][accession].append(float(p))
    return pred

if __name__ == "__main__":
    main()