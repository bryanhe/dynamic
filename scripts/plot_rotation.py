#!/usr/bin/env python3

"""Code to generate plots for Extended Data Fig. 3."""

import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import echonet

r2 = [
    (0.791, 0.767, 0.813),
    (0.810, 0.786, 0.831),
    (0.801, 0.778, 0.822),
    (0.807, 0.784, 0.828),
    (0.806, 0.782, 0.827),
    (0.798, 0.775, 0.819),
]

auc = [
    (0.840958605664488,  0.7598116169544741, 0.9122989102231448),
    (0.7979302832244008, 0.712241653418124, 0.8765840220385676),
    (0.8349673202614379, 0.7582417582417582, 0.9040431266846362),
    (0.8540305010893245, 0.7741239892183289, 0.9256933542647828),
    (0.8050108932461874, 0.7175202156334232, 0.8844866071428572),
    (0.8485838779956426, 0.7712418300653595, 0.9172113289760349),
]

r2 = np.array(r2)
auc = np.array(auc)

rotation = [0, 10, 20, 30, 40, 50]

echonet.utils.latexify()
fig = plt.figure(figsize=(3, 3))
# plt.errorbar(rotation, r2[:, 0], r2[:, 1:].transpose() - np.vstack((r2[:, 0], r2[:, 0])))
plt.errorbar(rotation, r2[:, 0], r2[:, 1] - r2[:, 0])
plt.title("Cardiology Videos")
plt.xlabel("Maximum Rotation")
plt.ylabel("R2")
plt.tight_layout()
plt.savefig("cardiology.pdf")

fig = plt.figure(figsize=(3, 3))
# plt.errorbar(rotation, r2[:, 0], r2[:, 1:].transpose() - np.vstack((r2[:, 0], r2[:, 0])))
plt.errorbar(rotation, auc[:, 0], auc[:, 1] - auc[:, 0])
plt.title("ER Videos")
plt.xlabel("Maximum Rotation")
plt.ylabel("AUC")
plt.tight_layout()
plt.savefig("er.pdf")


exit()
def main():
    """Generate plots for Extended Data Fig. 3."""

    # Select paths and hyperparameter to plot
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", nargs="?", default="output")
    parser.add_argument("fig", nargs="?", default=os.path.join("figure", "loss"))
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--period", type=int, default=2)
    args = parser.parse_args()

    # Set up figure
    echonet.utils.latexify()
    os.makedirs(args.fig, exist_ok=True)
    fig = plt.figure(figsize=(7, 5))
    gs = matplotlib.gridspec.GridSpec(ncols=3, nrows=2, figure=fig, width_ratios=[2.75, 2.75, 1.50])

    # Plot EF loss curve
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    for pretrained in [True]:
        for (model, color) in zip(["r2plus1d_18", "r3d_18", "mc3_18"], matplotlib.colors.TABLEAU_COLORS):
            loss = load(os.path.join(args.dir, "video", "{}_{}_{}_{}".format(model, args.frames, args.period, "pretrained" if pretrained else "random"), "log.csv"))
            ax0.plot(range(1, 1 + len(loss["train"])), loss["train"], "-" if pretrained else "--", color=color)
            ax1.plot(range(1, 1 + len(loss["val"])), loss["val"], "-" if pretrained else "--", color=color)

    plt.axis([0, max(len(loss["train"]), len(loss["val"])), 0, max(max(loss["train"]), max(loss["val"]))])
    ax0.text(-0.25, 1.00, "(a)", transform=ax0.transAxes)
    ax1.text(-0.25, 1.00, "(b)", transform=ax1.transAxes)
    ax0.set_xlabel("Epochs")
    ax1.set_xlabel("Epochs")
    ax0.set_xticks([0, 15, 30, 45])
    ax1.set_xticks([0, 15, 30, 45])
    ax0.set_ylabel("Training MSE Loss")
    ax1.set_ylabel("Validation MSE Loss")

    # Plot segmentation loss curve
    ax0 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[1, 1], sharey=ax0)
    pretrained = False
    for (model, color) in zip(["deeplabv3_resnet50"], list(matplotlib.colors.TABLEAU_COLORS)[3:]):
        loss = load(os.path.join(args.dir, "segmentation", "{}_{}".format(model, "pretrained" if pretrained else "random"), "log.csv"))
        ax0.plot(range(1, 1 + len(loss["train"])), loss["train"], "--", color=color)
        ax1.plot(range(1, 1 + len(loss["val"])), loss["val"], "--", color=color)

    ax0.text(-0.25, 1.00, "(c)", transform=ax0.transAxes)
    ax1.text(-0.25, 1.00, "(d)", transform=ax1.transAxes)
    ax0.set_ylim([0, 0.13])
    ax0.set_xlabel("Epochs")
    ax1.set_xlabel("Epochs")
    ax0.set_xticks([0, 25, 50])
    ax1.set_xticks([0, 25, 50])
    ax0.set_ylabel("Training Cross Entropy Loss")
    ax1.set_ylabel("Validation Cross Entropy Loss")

    # Legend
    ax = fig.add_subplot(gs[:, 2])
    for (model, color) in zip(["EchoNet-Dynamic (EF)", "R3D", "MC3", "EchoNet-Dynamic (Seg)"], matplotlib.colors.TABLEAU_COLORS):
        ax.plot([float("nan")], [float("nan")], "-", color=color, label=model)
    ax.set_title("")
    ax.axis("off")
    ax.legend(loc="center")

    plt.tight_layout()
    plt.savefig(os.path.join(args.fig, "loss.pdf"))
    plt.savefig(os.path.join(args.fig, "loss.eps"))
    plt.savefig(os.path.join(args.fig, "loss.png"))
    plt.close(fig)


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
