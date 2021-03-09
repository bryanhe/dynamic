#!/usr/bin/env python3

import os
import math
import numpy as np
from matplotlib import pyplot as plt

def main():
    os.makedirs(os.path.join("fig", "loss"), exist_ok=True)
    n_patients = [100, 250, 500, 1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 93739]
    # n_patients = [1000, 5000, 10000, 93739]
    r2 = []
    for n in n_patients:
        r2.append(load_test(os.path.join("output", "train_{}".format(None if n == 93739 else n), "log.csv")))
        
        loss = load(os.path.join("output", "train_{}".format(None if n == 93739 else n), "log.csv"))
        
        fig = plt.figure(figsize=(3, 3))
        plt.plot(range(1, 1 + len(loss["train"])), loss["train"], "-", label="Training")
        plt.plot(range(1, 1 + len(loss["val"])), loss["val"], "--", label="Validation")
        plt.ylim([0, 100])
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join("fig", "loss", "loss_{}.pdf".format(n)))
        plt.close(fig)
    r2 = np.array(r2)
    print(r2)
    r2[:, 1] -= r2[:, 0]
    r2[:, 1] *= -1
    r2[:, 2] -= r2[:, 0]

    fig = plt.figure(figsize=(3, 3))
    plt.errorbar(n_patients, r2[:, 0], r2[:, 1:].transpose())
    plt.xlabel("# Training Patients")
    plt.ylabel("$\mathrm{R}^2$")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join("fig", "train_size.pdf"))
    plt.close(fig)


def load_test(filename):
    r2 = math.nan
    lower = math.nan
    upper = math.nan
    with open(filename, "r") as f:
        for l in f:
            if "test (one clip) R2:" in l:
                print(l)
                *_, m, l, _, u = l.split()

                l = l[1:]
                u = u[:-1]

                r2 = float(m)
                lower = float(l)
                upper = float(u)

    return r2, lower, upper


def load(filename):
    """Loads losses from specified file."""

    losses = {"train": [], "val": []}
    with open(filename, "r") as f:
        for line in f:
            line = line.split(",")
            if len(line) < 4:
                continue
            epoch, split, loss, *_ = line
            epoch = int(epoch)
            loss = float(loss)
            assert(split in ["train", "val"])
            if epoch == len(losses[split]):
                losses[split].append(loss)
            elif epoch == len(losses[split]) - 1:
                losses[split][-1] = loss
            else:
                raise ValueError("File has uninterpretable formatting.")
    return losses


if __name__ == "__main__":
    main()
