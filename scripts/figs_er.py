#!/usr/bin/env python3

import concurrent.futures
import PIL
import click
import cv2
import numpy as np
import os
import tqdm
import skimage.segmentation

import echonet


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False))
@click.argument("dest", type=click.Path(file_okay=False))
def main(src, dest):
    os.makedirs(dest, exist_ok=True)

    video = "VID25908.webm"
    v = echonet.utils.loadvideo(os.path.join(src, video))
    PIL.Image.fromarray(v[:, 0, :, :].transpose((1, 2, 0))).save(os.path.join(dest, "example.jpg"))

    for (quartile, video) in enumerate([
        "VID35145.webm",
        "VID15746.webm", # "VID49072.webm",
        "VID44946.webm",
        # "VID25824.webm", # "VID44706.webm", # "VID43688.webm", # "VID52826.webm", # "VID41896.webm", # "VID46109.webm",
        "VID34740.webm", # "VID18443.webm", "VID39715.webm", "VID39893.webm",
        "VID25908.webm"
    ]):
        v = echonet.utils.loadvideo(os.path.join(src, video))
        PIL.Image.fromarray(v[:, 0, :, :].transpose((1, 2, 0))).save(os.path.join(dest, "quartile_{}.jpg".format(quartile)))


if __name__ == "__main__":
    main()
