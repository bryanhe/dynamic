#!/usr/bin/env python3

import torch
import torchvision
import echonet
import os
import tqdm
import numpy as np

def main():
    block_size = 1024
    output = "frame"
    os.makedirs(os.path.join(output, "trace"), exist_ok=True)

    device = torch.device("cuda")
    # Set up Model
    model = torchvision.models.segmentation.deeplabv3_resnet50(aux_loss=False)

    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)  # change number of outputs to 1
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    checkpoint = torch.load("deeplabv3_resnet50_random.pt")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Compute dataset statistics
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))

    # Set up dataloader
    def collate_fn(x):
        """Collate function for Pytorch dataloader to merge multiple videos.

        The dataset is expected to return a tuple containing the video, along with some (non-zero) additional features.
        This function returns a 3-tuple:
            - Videos concatenated along the frames dimension
            - The additional features converted into matrix form
            - Lengths of the videos
        """
        x, f = zip(*x)  # Split the input into the videos and the additional features
        f = zip(*f)  # Swap features back into original form (this and the previous line essentially do two transposes)
        i = list(map(lambda t: t.shape[1], x))  # Create a list of videos lengths (in frames)
        x = torch.as_tensor(np.swapaxes(np.concatenate(x, 1), 0, 1))  # Concatenate along frames dimension, and swap to be first dimension
        return x, f, i

    ds = echonet.datasets.Echo(split="all", target_type=["Filename", "LargeIndex", "SmallIndex"], mean=mean, std=std, length=None, max_length=None, period=1)
    done = set([os.path.splitext(x)[0] for x in os.listdir(os.path.join(output, "trace"))])
    mask = [os.path.splitext(x)[0] not in done for x in ds.fnames]
    ds.fnames = [x for (x, m) in zip(ds.fnames, mask) if m]
    ds.outcome = [x for (x, m) in zip(ds.outcome, mask) if m]
    # ds = echonet.datasets.Echo(split="test", target_type=["Filename", "LargeIndex", "SmallIndex"], mean=mean, std=std, length=None, max_length=None, period=1)
    dataloader = torch.utils.data.DataLoader(ds,
                                             batch_size=10, num_workers=5, shuffle=False, pin_memory=False, collate_fn=collate_fn)

    with torch.no_grad():
        for (x, (filenames, large_index, small_index), length) in tqdm.tqdm(dataloader):
            print(x.shape, flush=True)
            # Predict segmentation in blocks (only send subset of frames to gpu at a time)
            y = np.concatenate([model(x[i:(i + block_size), :, :, :].to(device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], block_size)])
            start = 0
            for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                # Extract one video and segmentation predictions
                logit = y[start:(start + offset), 0, :, :]
                np.save(os.path.join(output, "trace", os.path.splitext(filename)[0] + ".npy"), logit)
                start += offset

if __name__ == "__main__":
    main()
