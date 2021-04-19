#!/usr/bin/env python3

import os

import click
import pandas
import sklearn
import matplotlib.pyplot as plt
import numpy as np

import echonet


@click.command()
@click.argument("prediction", type=click.File("r"))
@click.argument("labels", type=click.File("r"))
def main(prediction, labels):
    pred = pandas.read_csv(prediction, header=None, names=["VID#", "EF"])
    label = pandas.read_csv(labels)
    # pred["VID#"] = pred["VID#"].apply(lambda x: x[3:-4]).astype(label["VID#"].dtype)
    pred["VID#"] = pred["VID#"].apply(lambda x: x[3:-4]).astype(int)

    pred = pred.set_index("VID#")
    label = label.set_index("VID#")

    print(pred)
    print(label)

    data = pred.join(label)

    print(data)

    # File sorted by EF pred
    data.sort_values("EF").to_csv("predictions.csv")


    assert set(data["EF normal vs low"].unique()) == set(["normal", "low"])
    assert set(data["Clinically interpretable - difficult vs not"].unique() == "not") == set([True, False])

    data["EF normal vs low"] = (data["EF normal vs low"] == "normal")

    tasks = [
        ("All", np.ones(data.shape[0], dtype=np.bool)),
        ("Interpretable", data["Clinically interpretable - difficult vs not"] == "not"),
        ("Difficult", data["Clinically interpretable - difficult vs not"] == "diff"),
        ("GE venue", data["Machine"] == "GE venue"),
        ("TE7", data["Machine"] == "TE7"),
        # ("Machine (Phillips)", data["Machine"] == "Phillips"),
    ]

    os.makedirs("fig", exist_ok=True)
    for (name, mask) in tasks:
        fpr, tpr, _ = sklearn.metrics.roc_curve(data[mask]["EF normal vs low"], data[mask]["EF"])
        fig = plt.figure(figsize=(3.5, 3.5))
        plt.plot(fpr, tpr)
        plt.axis([0, 1, 0, 1])
        plt.title("{} (AUROC={:.2f}, n={})".format(name, sklearn.metrics.roc_auc_score(data[mask]["EF normal vs low"], data[mask]["EF"]), mask.sum()))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.tight_layout()
        plt.savefig(os.path.join("fig", name.lower().replace(" ", "_",).replace("(", "").replace(")", "") + ".pdf"))
        plt.close(fig)

    # Violin plot
    fig = plt.figure(figsize=(3, 3))
    plt.hist([data[data["EF normal vs low"]]["EF"], data[~data["EF normal vs low"]]["EF"]])
    plt.legend(["Normal", "Low"])
    plt.xlabel('Predicted EF')
    plt.ylabel('# of Videos')
    plt.tight_layout()
    plt.savefig(os.path.join("fig", "hist.pdf"))
    plt.close(fig)



if __name__ == "__main__":
    main()
