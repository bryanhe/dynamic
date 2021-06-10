#!/usr/bin/env python3

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
import tqdm

import echonet


@click.command()
@click.argument("labels", type=click.Path(exists=True, file_okay=False))
@click.argument("videos", type=click.Path(exists=True, file_okay=False))
@click.argument("dest", type=click.Path(file_okay=False))
def main(labels, videos, dest, splits=10):

    try:
        with open(os.path.join(dest, "exams.pkl"), "rb") as f:
            exam = pickle.load(f)
    except FileNotFoundError:
        exam = {}
        for e in tqdm.tqdm(os.listdir("data/ER_Exams")):
            if e == "[Content_Types].xml":
                continue
            with zipfile.ZipFile(os.path.join("data", "ER_Exams", e)) as zf:
                if "MPEG.zip" not in zf.namelist():
                    print(e, "does not have MPEG.zip")
                    continue
                with zf.open("MPEG.zip") as f:
                    data = io.BytesIO(f.read())  # https://stackoverflow.com/questions/11930515/unzip-nested-zip-files-in-python
                    with zipfile.ZipFile(data) as inner:
                        e = e.split("-")[0][4:]
                        assert e not in exam
                        exam[e] = list(filter(lambda x: os.path.splitext(x)[-1] == ".mp4", inner.namelist()))
        with open(os.path.join(dest, "exams.pkl"), "wb") as f:
            pickle.dump(exam, f)

    assert len(sum(exam.values(), [])) == len(set(sum(exam.values(), [])))

    video_to_exam = {}
    for e in exam:
        for v in exam[e]:
            video_to_exam[os.path.splitext(v)[0]] = e

    assert all(os.path.splitext(v)[0] in video_to_exam for v in os.listdir(videos))

    # breakpoint()
    # >>> len(list(map(lambda x: video_to_exam[os.path.splitext(x)[0]], os.listdir(videos))))
    # >>> len(set(map(lambda x: video_to_exam[os.path.splitext(x)[0]], os.listdir(videos))))

    os.makedirs(os.path.join(dest, "Videos"), exist_ok=True)
    for filename in tqdm.tqdm(os.listdir(videos)):
        if not os.path.isfile(os.path.join(dest, "Videos", os.path.splitext(filename)[0] + ".avi")):
            video = echonet.utils.loadvideo(os.path.join(videos, filename))
            video = video[:, :, :, ((video.shape[3] - video.shape[2]) // 2):((video.shape[3] - video.shape[2]) // 2) + video.shape[2]]
            size = (112, 112)
            video = np.array(list(map(lambda x: cv2.resize(x, size, interpolation=cv2.INTER_AREA), video.transpose((1, 2, 3, 0))))).transpose((3, 0, 1, 2))
            echonet.utils.savevideo(os.path.join(dest, "Videos", os.path.splitext(filename)[0] + ".avi"), video, fps=50)

    label = {}
    for annotator in ["DD", "TT", "Consensus"]:
        label[annotator] = {}
        for pkl_name in os.listdir(os.path.join(labels, annotator)):
            with open(os.path.join(labels, annotator, pkl_name), "rb") as f:
                data = pickle.load(f)
                if "EF" in data and "Interpretable" in data:
                    filename = os.path.splitext(pkl_name)[0]
                    if not os.path.isfile(os.path.join(dest, "Videos", filename + ".avi")):
                        print("{} is not parseable.".format(filename))
                        continue
                    label[annotator][pkl_name] = (data["EF"], data["Interpretable"])

    ef_score = {
        "Normal": 3,
        "Slightly Reduced": 2,
        "Moderately Reduced": 1,
        "Severely Reduced": 0,
    }
    interpret_score = {
        "Yes": 2,
        "Partial": 1,
        "No": 0,
    }

    consensus = {}
    for pkl_name in label["TT"]:
        if pkl_name not in label["Consensus"]:
            assert abs(ef_score[label["TT"][pkl_name][0]] - ef_score[label["DD"][pkl_name][0]]) < 2
            assert sorted((ef_score[label["TT"][pkl_name][0]], ef_score[label["DD"][pkl_name][0]])) != [1, 2]
            assert abs(interpret_score[label["TT"][pkl_name][1]] - interpret_score[label["DD"][pkl_name][1]]) < 2
            consensus[pkl_name] = (label["TT"][pkl_name][0], interpret_score[label["TT"][pkl_name][1]] + interpret_score[label["TT"][pkl_name][1]])
        else:
            ef = label["TT"][pkl_name][0]
            if abs(ef_score[label["TT"][pkl_name][0]] - ef_score[label["DD"][pkl_name][0]]) >= 2 or sorted((ef_score[label["TT"][pkl_name][0]], ef_score[label["DD"][pkl_name][0]])) == [1, 2]:
                lower, upper = sorted([ef_score[label["TT"][pkl_name][0]], ef_score[label["DD"][pkl_name][0]]])
                if not (lower <= ef_score[label["Consensus"][pkl_name][0]] <= upper):
                    print("EF (consensus)")
                    print(pkl_name)
                    print(label["TT"][pkl_name])
                    print(label["DD"][pkl_name])
                    print(label["Consensus"][pkl_name])
                    print()
                ef = label["Consensus"][pkl_name][0]
            interpret = interpret_score[label["TT"][pkl_name][1]] + interpret_score[label["TT"][pkl_name][1]]
            if abs(interpret_score[label["TT"][pkl_name][1]] - interpret_score[label["DD"][pkl_name][1]]) >= 2:
                lower, upper = sorted([interpret_score[label["TT"][pkl_name][1]], interpret_score[label["DD"][pkl_name][1]]])
                if not (lower <= interpret_score[label["Consensus"][pkl_name][1]] <= upper):
                    print("Interpretable (consensus)")
                    print(pkl_name)
                    print(label["TT"][pkl_name])
                    print(label["DD"][pkl_name])
                    print(label["Consensus"][pkl_name])
                    print()
                interpret = 1 + interpret_score[label["Consensus"][pkl_name][1]]
            consensus[pkl_name] = (ef, interpret)

            # print(pkl_name)
            # print(label["TT"][pkl_name])
            # print(label["DD"][pkl_name])
            # print(label["Consensus"][pkl_name])
            # print(consensus[pkl_name])
            # print()

    ef = collections.Counter()
    interpretable = collections.Counter()
    for pkl_name in sorted(label["TT"]):
        if pkl_name in label["DD"] and pkl_name in label["TT"]:
            if label["DD"][pkl_name][0] == "Severely Reduced" and label["TT"][pkl_name][0] == "Normal":
                print(pkl_name)
            ef[label["DD"][pkl_name][0], label["TT"][pkl_name][0]] += 1
            interpretable[label["DD"][pkl_name][1], label["TT"][pkl_name][1]] += 1


    print("ef")
    for i in ["Normal", "Slightly Reduced", "Moderately Reduced", "Severely Reduced"]:
        for j in ["Normal", "Slightly Reduced", "Moderately Reduced", "Severely Reduced"]:
            print(ef[i, j], end="\t")
        print()

    print("interpretable")
    for i in ["Yes", "Partial", "No"]:
        for j in ["Yes", "Partial", "No"]:
            print(interpretable[i, j], end="\t")
        print()

    files = []
    assert "TT" in os.listdir(labels)
    for pkl_name in consensus:
        filename = os.path.splitext(pkl_name)[0]
        assert os.path.isfile(os.path.join(dest, "Videos", filename + ".avi"))
        files.append((filename + ".avi", consensus[pkl_name][0], consensus[pkl_name][1], int(hashlib.sha1(video_to_exam[filename].encode("utf-8")).hexdigest(), 16) % 10))

    _, ef, interpretable, _ = zip(*files)

    print(collections.Counter(e for (e, i) in zip(ef, interpretable) if i != "No"))
    print(collections.Counter(interpretable))

    with open(os.path.join(dest, "FileList.csv"), "w") as f:
        f.write("FileName,EF,Interpretable,Split\n")
        for (filename, ef, interpretable, s) in files:
            f.write("{},{},{}\n".format(filename, ef, interpretable))

    breakpoint()
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
                f.write("{},{},{},{}\n".format(filename, 1 if ef == "Normal" or ef == "Slightly Reduced" else 0, 0 if interpretable <= 1 else 1, s))

if __name__ == "__main__":
    main()
