#!/usr/bin/env python3

import click
import cv2
import numpy as np
import os
import tqdm

import echonet


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False))
@click.argument("dest", type=click.Path(file_okay=False))
def main(src, dest):
    os.makedirs(dest, exist_ok=True)
    for filename in tqdm.tqdm(sorted(os.listdir(src))):
        # if filename in ["VID11216.mp4", "VID11713.mp4", "VID14180.mp4", "VID14278.mp4", "VID14441.mp4", "VID1480.mp4", "VID1481.mp4", "VID16642.mp4", "VID16885.mp4"]:
        #     continue
        # output = os.path.join(dest, os.path.splitext(filename)[0] + ".avi")
        output = os.path.join(dest, filename)
        if not os.path.isfile(output):
            print(filename)
            capture = cv2.VideoCapture(os.path.join(src, filename))
            fps = capture.get(cv2.CAP_PROP_FPS)
            print(fps)
            video = echonet.utils.loadvideo(os.path.join(src, filename))
            print(video.shape)
            video = crop(video)
            video = resize(video)
            video = mask(video)
            echonet.utils.savevideo(output, video, fps)

def crop(video):
    h = video.shape[2]
    w = video.shape[3]
    start = round(0.6 * (w - h))
    return video[:, :, :, start:(start + h)]

def resize(video, size=(112, 112)):
    return np.array(list(map(lambda x: cv2.resize(x, size, interpolation=cv2.INTER_AREA), video.transpose((1, 2, 3, 0))))).transpose((3, 0, 1, 2))

def mask(video):
    h = video.shape[2]
    w = video.shape[3]

    target = np.any(video > 10, (0, 1))

    n = h
    x, y = np.meshgrid(np.arange(n), np.arange(n))

    best = None
    n = 25
    for i in range(-n, n + 1, 2):
        for j in range(0, n + 1, 2):
            x_offset = w // 2 + i
            y_offset = j

            mask = np.logical_and(y - y_offset > x - w // 2 - i,
                                  y - y_offset > w - x - w // 2 + i)
            window = np.logical_and.reduce((
                    x_offset - 100 < x,
                    x < x_offset + 100,
                    y_offset < y,
                    y < y_offset + 100))

            score = target[np.logical_and(window, mask)].sum() / np.logical_and(window, mask).sum() + (~target[np.logical_and(window, ~mask)]).sum() / np.logical_and(window, ~mask).sum()
            score = (score, i, j)
            if best is None or score > best:
                best = score
                print(best)

    _, i, j = best
    x_offset = w // 2 + i
    y_offset = j
    mask = np.logical_and(y - y_offset > x - w // 2 - i,
                          y - y_offset > w - x - w // 2 + i)
    window = np.logical_and.reduce((
            x_offset - 100 < x,
            x < x_offset + 100,
            y_offset < y,
            y < y_offset + 100))

    #video[1, :] = (target * 255).astype(np.uint8)
    # video[2, :] = (window * 255).astype(np.uint8)
    # video[0, :, ~mask] = 255

    video[:, :, ~mask] = 0
    return video

if __name__ == "__main__":
    main()
