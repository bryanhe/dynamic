#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
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

    try:
        return sklearn.metrics.r2_score(a, b), math.sqrt(sklearn.metrics.mean_squared_error(a, b))
    except:
        return math.nan, math.nan

def bootstrap(ef, pred, n=100):
    point = score(pred.keys(), ef, pred)

    patients = list(pred.keys())
    samples = []
    for _ in range(n):
        samples.append(score([random.choice(patients) for _ in range(len(patients))], ef, pred))

    samples = list(map(sorted, zip(*samples)))
    return [(p, s[round(0.025 * len(s))], s[round(0.975 * len(s))]) for (p, s) in zip(point, samples)]


def main(src="output/pediatric/ef"):
    ef = collections.defaultdict(dict)
    sex = {}
    age = collections.defaultdict(dict)
    height = collections.defaultdict(dict)
    weight = collections.defaultdict(dict)
    for view in ["A4C", "PSAX"]:
        with open("data/pediatric/{}/FileList.csv".format(view)) as f:
            assert f.readline() == "FileName,EF,Sex,Age,Weight,Height,Split\n"

            for line in f:
                filename, e, s, a, w, h, _split = line.split(",")
                patient, accession, _ = os.path.splitext(filename)[0].split("-")
                e = float(e)

                if accession in ef[patient]:
                    assert ef[patient][accession] == e
                ef[patient][accession] = e
                if patient in sex:
                    pass
                    # assert s in "MF"
                    # assert sex[patient] == s
                sex[patient] = s
                # TODO: check unique
                age[patient][accession] = a  # TODO: check that age makes sense
                weight[patient][accession] = w
                height[patient][accession] = h
                sex[patient] = s


    for method in [
        "lr_1e-4",
        "blind",
        "scratch",
        "lr_1e-4",
        # "lr_1e-5",
        # "lr_1e-6",
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
        print("Visits:", sum(map(len, ef.values())))  # TODO: check same as merged?
        for (score, (p, l, h)) in zip(["R2", "RMSE"], bootstrap(ef, merged)):
            print("{}: {:.2f} ({:.2f} - {:.2f})".format(score, p, l, h))
        print()
        fig = plt.figure(figsize=(3, 3))
        plt.scatter(*zip(*[(ef[p][a], merged[p][a]) for p in ef for a in ef[p] if p in merged and a in merged[p]]), s=1, color="k")
        plt.xlabel("Real EF")
        plt.ylabel("Predicted EF")
        plt.savefig(method + ".pdf")
        plt.tight_layout()
        plt.close(fig)

        for (group, mask) in [
            ("Low EF", {(p, a) for p in ef for a in ef[p] if ef[p][a] < 60}),
            ("Normal EF", {(p, a) for p in ef for a in ef[p] if ef[p][a] >= 60}),
            ("Male", {(p, a) for p in ef for a in ef[p] if sex[p] == "M"}),
            ("Female", {(p, a) for p in ef for a in ef[p] if sex[p] == "F"}),
            ("Age (<1)", {(p, a) for p in ef for a in ef[p] if int(age[p][a]) < 1}),
            ("Age (1-5)", {(p, a) for p in ef for a in ef[p] if 1 <= int(age[p][a]) <= 5}),
            ("Age (6-11)", {(p, a) for p in ef for a in ef[p] if 6 <= int(age[p][a]) <= 11}),
            ("Age (12-14)", {(p, a) for p in ef for a in ef[p] if 12 <= int(age[p][a]) <= 14}),
            ("Age (>=15)", {(p, a) for p in ef for a in ef[p] if 15 <= int(age[p][a])}),
        ]:
            print(group)
            # TODO: number of visits

            ef_mask = {p: {a: ef[p][a] for a in ef[p] if (p, a) in mask} for p in ef}
            ef_mask = {p: ef_mask[p] for p in ef if ef_mask[p] != {}}

            merged_mask = {p: {a: merged[p][a] for a in merged[p] if (p, a) in mask} for p in merged}
            merged_mask = {p: merged_mask[p] for p in merged if merged_mask[p] != {}}
            print("Visits:", sum(map(len, ef_mask.values())))
            print("EF:", 
                  np.mean(sum(map(lambda x: list(x.values()), ef_mask.values()), [])),
                  "+/-",
                  np.std(sum(map(lambda x: list(x.values()), ef_mask.values()), [])),
                  )


            for (score, (p, l, h)) in zip(["R2", "RMSE"], bootstrap(ef_mask, merged_mask)):
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
