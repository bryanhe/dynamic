#!/usr/bin/env python3

import skimage.feature
import hashlib
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
import collections

import echonet


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False))
@click.argument("dest", type=click.Path(file_okay=False))
@click.argument("patients", nargs=-1)
def main(src, dest, patients):
    try:
        root = {}
        with open(os.path.join(dest, "root.tsv"), "r") as f:
            for line in f:
                p, *path = line.strip().split("\t")
                root[p] = path
    except FileNotFoundError:
        root = {}
        for batch in os.listdir(src):
            if batch in ["$RECYCLE.BIN", "System Volume Information"]:
                continue

            for patient in os.listdir(os.path.join(src, batch)):
                if not os.path.isdir(os.path.join(src, batch, patient)) or patient == "Viewer":
                    continue

                # print(batch, patient)
                p = patient.split("_")[0]
                assert patient == "{}_000_{}_".format(p, p) or patient == "{}_{}_".format(p, p)
                if p in root:
                    # Patient appears multiple times
                    # (probably across batches, but potentially just with missing _000)
                    print(root[p])
                    print(batch, patient)

                    prev = collections.defaultdict(int)
                    for filename in os.listdir(os.path.join(src, root[p])):
                        prev[filename] = os.path.getsize(os.path.join(src, root[p], filename))

                    curr = collections.defaultdict(int)
                    for filename in os.listdir(os.path.join(src, batch, patient)):
                        curr[filename] = os.path.getsize(os.path.join(src, batch, patient, filename))

                    filenames = set().union(prev.keys(), curr.keys())

                    if all(prev[f] == curr[f] for f in filenames):
                        # print("Same")
                        pass
                    elif all(prev[f] >= curr[f] for f in filenames):
                        # print("Old is bigger")
                        pass
                    elif all(prev[f] <= curr[f] for f in filenames):
                        # print("New is bigger")
                        root[p] = os.path.join(batch, patient)
                    else:
                        print("Mixed")
                        assert p == "90084209"
                        # if len(os.listdir(os.path.join(src, batch, patient))) > len(os.listdir(os.path.join(src, *root[p]))):
                        root[p] = [root[p], os.path.join(batch, patient)]

                    print()

                else:
                    root[p] = os.path.join(batch, patient)

        for p in sorted(root):
            if not isinstance(root[p], list):
                root[p] = [root[p]]

        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, "root.tsv"), "w") as f:
            for p in sorted(root):
                f.write("\t".join((p, *root[p])) + "\n")

    logo = PIL.Image.open(os.path.join(os.path.dirname(__file__), "phillips.png"))
    logo = np.array(logo)

    if patients == ():
        patients = sorted(root.keys())
    print(patients)

    with tqdm.tqdm(total=len(patients)) as pbar:
        for p in patients:
            pbar.set_postfix(patient=p)

            def save_file(filename):
                r, filename = filename
                if os.path.isfile(os.path.join(dest, "Videos", p, "full", filename[3:] + ".webm")) or \
                   os.path.isfile(os.path.join(dest, "Videos", p, "color", filename[3:] + ".webm")):
                   # skip if done
                   return

                ds = pydicom.dcmread(os.path.join(src, r, filename), force=True)

                try:
                    video = ds.pixel_array
                except AttributeError:
                    return
                if len(video.shape) in (2, 3):
                    return

                res = skimage.feature.match_template(video[0, :, :, :], logo)
                i, j, _ = np.unravel_index(np.argmax(res), res.shape)

                try:
                    if len(ds.SequenceOfUltrasoundRegions) != 1:
                        print("Found {} regions; expected 1.".format(len(ds.SequenceOfUltrasoundRegions)))
                except AttributeError:
                    # TODO: why do some miss this?
                    return

                # assert len(ds.SequenceOfUltrasoundRegions) == 1
                region = ds.SequenceOfUltrasoundRegions[0]
                x0 = region.RegionLocationMinX0
                y0 = region.RegionLocationMinY0
                x1 = region.RegionLocationMaxX1
                y1 = region.RegionLocationMaxY1

                video = pydicom.pixel_data_handlers.util.convert_color_space(video, ds.PhotometricInterpretation, "RGB")
                video = video.transpose((3, 0, 1, 2))

                small = video[:, :, y0:(y1 + 1), x0:(x1 + 1)]
                _, _, h, w = small.shape
                small = small[:, :, :, ((w - h) // 2):(h + (w - h) // 2)]
                small = np.array(list(map(lambda x: cv2.resize(x, (112, 112), interpolation=cv2.INTER_AREA), small.transpose((1, 2, 3, 0))))).transpose((3, 0, 1, 2))

                if i > 250:
                    pass
                    # Upside-down
                    # video = video[:, :, ::-1, :]
                    # small = small[:, :, ::-1, :]

                try:
                    fps = 1000 / float(ds.FrameTime)
                except AttributeError:
                    return

                assert filename[:3] == "IMG"

                try:
                    color = (ds.UltrasoundColorDataPresent != 0)
                except AttributeError:
                    return

                if not color:
                    echonet.utils.savevideo(os.path.join(dest, "Videos", p, filename[3:] + ".webm"), small, fps)

                    os.makedirs(os.path.join(dest, "Videos", p, "full"), exist_ok=True)
                    echonet.utils.savevideo(os.path.join(dest, "Videos", p, "full", filename[3:] + ".webm"), video, fps)
                else:
                    os.makedirs(os.path.join(dest, "Videos", p, "color"), exist_ok=True)
                    echonet.utils.savevideo(os.path.join(dest, "Videos", p, "color", filename[3:] + ".webm"), video, fps)

            filenames = []
            for r in root[p]:
                filenames.extend((r, f) for f in os.listdir(os.path.join(src, r)))
            filenames = sorted(filenames)

            if not os.path.isfile(os.path.join(dest, "Videos", p, "complete.txt")):
                os.makedirs(os.path.join(dest, "Videos", p), exist_ok=True)
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    for filename in tqdm.tqdm(executor.map(save_file, filenames), total=len(filenames), leave=False):
                    # for filename in tqdm.tqdm(map(save_file, filenames), total=len(filenames), leave=False):
                        pass

                with open(os.path.join(dest, "Videos", p, "complete.txt"), "w") as f:
                    # write file to mark complete
                    f.write(p + "\n")

            pbar.update()


if __name__ == "__main__":
    main()
