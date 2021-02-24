#!/usr/bin/env python3

import click
import torch
import torchvision
import tqdm
import os
import numpy as np
import scipy
import skimage
import matplotlib.pyplot as plt
import cv2

import echonet


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False))
@click.argument("dest", type=click.Path(file_okay=False))
@click.option("--weights", type=click.Path(exists=True, file_okay=False), default="model")
def main(src, dest, weights):

    # Initialize and Run EF model
    frames = 32
    period = 2
    batch_size = 20
    block_size = 256
    device = torch.device("cuda")

    model = torchvision.models.video.r2plus1d_18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model = torch.nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load(os.path.join(weights, "r2plus1d_18_32_2_pretrained.pt"))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    ds = echonet.datasets.Echo(split="external_test", external_test_location=src, clips=1)
    print(ds.split, ds.fnames)

    mean, std = echonet.utils.get_mean_and_std(ds)

    kwargs = {
        "target_type": "EF",
        "mean": mean,
        "std": std,
        "length": frames,
        "period": period,
    }

    # ds = echonet.datasets.Echo(split="external_test", external_test_location=src, **kwargs, clips="all")
    ds = echonet.datasets.Echo(split="external_test", external_test_location=src, **kwargs, clips=5)

    test_dataloader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=5, shuffle=False, pin_memory=(device.type=="cuda"))
    loss, yhat, y = echonet.utils.video.run_epoch(model, test_dataloader, False, None, device, save_all=False, block_size=25)

    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "predictions.csv"), "w") as f:
        for (filename, pred) in zip(ds.fnames, yhat):
            f.write("{},{}\n".format(filename, int(round(pred))))

    model = torchvision.models.segmentation.deeplabv3_resnet50(aux_loss=False)
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)  # change number of outputs to 1
    model = torch.nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load(os.path.join(weights, "deeplabv3_resnet50_random.pt"))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Saving videos with segmentations
    def collate_fn(x):
        """Collate function for Pytorch dataloader to merge multiple videos.

        The dataset is expected to return a tuple containing the video, along with some (non-zero) additional features.
        This function returns a 3-tuple:
            - Videos concatenated along the frames dimension
            - The additional features converted into matrix form
            - Lengths of the videos
        """
        x, f = zip(*x)  # Split the input into the videos and the additional features
        # f = zip(*f)  # Swap features back into original form (this and the previous line essentially do two transposes)
        i = list(map(lambda t: t.shape[1], x))  # Create a list of videos lengths (in frames)
        x = torch.as_tensor(np.swapaxes(np.concatenate(x, 1), 0, 1))  # Concatenate along frames dimension, and swap to be first dimension
        return x, f, i

    ds = echonet.datasets.Echo(split="external_test", external_test_location=src, target_type=["Filename"], mean=mean, std=std, length=None, max_length=None, period=1)
    dataloader = torch.utils.data.DataLoader(
            ds, batch_size=10, num_workers=0, shuffle=False, pin_memory=False, collate_fn=collate_fn)

    os.makedirs(os.path.join(dest, "videos"), exist_ok=True)
    os.makedirs(os.path.join(dest, "size"), exist_ok=True)
    echonet.utils.latexify()

    with torch.no_grad():
        with open(os.path.join(dest, "size.csv"), "w") as g:
            for (x, filenames, length) in tqdm.tqdm(dataloader):
                # Run segmentation model on blocks of frames one-by-one
                # The whole concatenated video may be too long to run together
                y = np.concatenate([model(x[i:(i + block_size), :, :, :].to(device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], block_size)])

                start = 0
                x = x.numpy()
                for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                    capture = cv2.VideoCapture(os.path.join(src, filename))
                    fps = capture.get(cv2.CAP_PROP_FPS)

                    # Extract one video and segmentation predictions
                    video = x[start:(start + offset), ...]
                    logit = y[start:(start + offset), 0, :, :]

                    # Un-normalize video
                    video *= std.reshape(1, 3, 1, 1)
                    video += mean.reshape(1, 3, 1, 1)

                    # Get frames, channels, height, and width
                    f, c, h, w = video.shape  # pylint: disable=W0612
                    assert c == 3

                    # Put two copies of the video side by side
                    video = np.concatenate((video, video), 3)

                    # If a pixel is in the segmentation, saturate blue channel
                    # Leave alone otherwise
                    video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])  # pylint: disable=E1111

                    # Add blank canvas under pair of videos
                    video = np.concatenate((video, np.zeros_like(video)), 2)

                    # Compute size of segmentation per frame
                    size = (logit > 0).sum((1, 2))

                    # Identify systole frames with peak detection
                    trim_min = sorted(size)[round(len(size) ** 0.05)]
                    trim_max = sorted(size)[round(len(size) ** 0.95)]
                    trim_range = trim_max - trim_min
                    systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])

                    # Plot sizes
                    fig = plt.figure(figsize=(size.shape[0] / fps * 1.5, 3))
                    plt.scatter(np.arange(size.shape[0]) / fps, size, s=1)
                    ylim = plt.ylim()
                    for s in systole:
                        plt.plot(np.array([s, s]) / fps, ylim, linewidth=1)
                    plt.ylim(ylim)
                    plt.title(os.path.splitext(filename)[0])
                    plt.xlabel("Seconds")
                    plt.ylabel("Size (pixels)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(dest, "size", os.path.splitext(filename)[0] + ".pdf"))
                    plt.close(fig)

                    # Normalize size to [0, 1]
                    size -= size.min()
                    size = size / size.max()
                    size = 1 - size

                    # Iterate the frames in this video
                    for (f, s) in enumerate(size):

                        # On all frames, mark a pixel for the size of the frame
                        video[:, :, int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))] = 255.

                        if f in systole:
                            # If frame is computer-selected systole, mark with a line
                            video[:, :, 115:224, int(round(f / len(size) * 200 + 10))] = 255.

                        def dash(start, stop, on=10, off=10):
                            buf = []
                            x = start
                            while x < stop:
                                buf.extend(range(x, x + on))
                                x += on
                                x += off
                            buf = np.array(buf)
                            buf = buf[buf < stop]
                            return buf
                        d = dash(115, 224)

                        # Get pixels for a circle centered on the pixel
                        r, c = skimage.draw.circle(int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10)), 4.1)

                        # On the frame that's being shown, put a circle over the pixel
                        video[f, :, r, c] = 255.

                    # Rearrange dimensions and save
                    video = video.transpose(1, 0, 2, 3)
                    video = video.astype(np.uint8)
                    echonet.utils.savevideo(os.path.join(dest, "videos", filename), video, fps)

                    # Move to next video
                    start += offset


if __name__ == "__main__":
    main()
