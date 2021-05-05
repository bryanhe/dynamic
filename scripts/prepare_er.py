#!/usr/bin/env python3

import pickle
import time
import matplotlib.pyplot as plt
import collections
import pydicom
import io
import re
import PIL
import click
import cv2
import numpy as np
import os
import tqdm
import skimage.segmentation
import zipfile
import tarfile
import concurrent.futures

import echonet


@click.command()
@click.argument("labels", type=click.Path(exists=True, file_okay=False))
@click.argument("videos", type=click.Path(exists=True, file_okay=False))
@click.argument("dest", type=click.Path(file_okay=False))
def main(labels, videos, dest, splits=10):
    os.makedirs(os.path.join(dest, "Videos"), exist_ok=True)
    for filename in tqdm.tqdm(os.listdir(videos)):
        if filename in [
            "VID43309.webm", "VID43300.webm", "VID43503.webm", "VID43308.webm", "VID43529.webm", "VID43644.webm", "VID43364.webm", "VID43527.webm", "VID43365.webm", "VID43587.webm", "VID43307.webm",
            "VID43645.webm" # zero-dim video
        ]:
            continue
        if not os.path.isfile(os.path.join(dest, "Videos", os.path.splitext(filename)[0] + ".avi")):
            video = echonet.utils.loadvideo(os.path.join(videos, filename))
            video = video[:, :, :, ((video.shape[3] - video.shape[2]) // 2):((video.shape[3] - video.shape[2]) // 2) + video.shape[2]]
            size = (112, 112)
            video = np.array(list(map(lambda x: cv2.resize(x, size, interpolation=cv2.INTER_AREA), video.transpose((1, 2, 3, 0))))).transpose((3, 0, 1, 2))
            echonet.utils.savevideo(os.path.join(dest, "Videos", os.path.splitext(filename)[0] + ".avi"), video, fps=50)

    files = []
    assert "TT" in os.listdir(labels)
    i = 0
    for pkl_name in os.listdir(os.path.join(labels, "TT")):
        with open(os.path.join(labels, "TT", pkl_name), "rb") as f:
            data = pickle.load(f)
            if "EF" in data and "Interpretable" in data:
                filename = os.path.splitext(pkl_name)[0]
                if not os.path.isfile(os.path.join(dest, "Videos", filename + ".avi")):
                    print("{} is not parseable.".format(filename))
                    continue
                files.append((filename + ".avi", data["EF"], data["Interpretable"], i % splits))
                i += 1

    for split in range(splits):
        os.makedirs(os.path.join(dest, "split_{}".format(split)), exist_ok=True)
        try:
            os.symlink(os.path.join("..", "Videos"), os.path.join(dest, "split_{}".format(split), "Videos"))
        except FileExistsError:
            pass
        with open(os.path.join(dest, "split_{}".format(split), "FileList.csv"), "w") as f:
            f.write("FileName,EF,Interpretable,Split\n")
            for (filename, ef, interpretable, s) in files:
                if s == split:
                    s = "TEST"
                elif (s + 1) % splits == split:
                    s = "VAL"
                else:
                    s = "TRAIN"
                f.write("{},{},{},{}\n".format(filename, 1 if ef == "Normal" else 0, 1 if interpretable != "No" else 0, s))

if __name__ == "__main__":
    main()
