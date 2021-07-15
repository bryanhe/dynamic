#!/usr/bin/env python3

import numpy as np
import echonet
import tqdm
import os
import torch
import torchvision
import click

@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False))
def main(src):
    device = torch.device("cuda") #use this GPU
    categories = ["a2c", "a4c", "plax", "subcostal", "other"]

    model = torchvision.models.video.r2plus1d_18(pretrained=False, num_classes=len(categories)) #r3d_18

    model = torch.nn.DataParallel(model)  # can use multiple GPUs
    model.to(device)  # move model to the GPUs

    checkpoint = torch.load("frames_CEloss_best_checkpoint_4ViewsWithOtherRound2.pt")
    model.load_state_dict(checkpoint["state_dict"])

    frames = checkpoint["frames"]
    period = checkpoint["period"]

    dataset = Echo(src, length=frames, period=period)

    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=16, 
                                             num_workers=8,  # number of subprocesses used for data loading. more subprocesses takes up more memory
                                             shuffle=False, 
                                             pin_memory=(device.type == "cuda"))

    os.makedirs(os.path.join(src, "view"), exist_ok=True)
    view_file = [open(os.path.join(src, "view", v + ".txt"), "w") for v in categories]
    for (x, filename) in tqdm.tqdm(dataloader):
        yhat = model(x)
        # view = [categories[i] for i in yhat.argmax(1)]
        view = yhat.argmax(1)

        for (f, v) in zip(filename, view):
            view_file[v].write(f + "\n")

        for f in view_file:
            f.flush()

    for f in view_file:
        f.close()

class Echo(torchvision.datasets.VisionDataset):
    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 max_clips=100,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None):
        super().__init__(root, target_transform=target_transform)

        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.max_clips = max_clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        patients = os.listdir(root)
        patients = sorted(p for p in patients if p not in ["root.tsv", "view"])

        videos = []
        for p in tqdm.tqdm(patients):
            for filename in os.listdir(os.path.join(root, p)):
                if os.path.splitext(filename)[-1] == ".webm":
                    videos.append(os.path.join(p, filename))

        self.fnames = videos

    def __getitem__(self, index):
        video = os.path.join(self.root, self.fnames[index])

        # Load video into np.array
        video = echonet.utils.loadvideo(video).astype(np.float32)

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
            if start.size > self.max_clips:
                # TODO: this messes up the clip number in test-time aug
                # Might need to have a clip index target
                start = np.random.choice(start, self.max_clips, replace=False)
                start.sort()
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Select clips from video
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        return video, self.fnames[index]

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)





if __name__ == "__main__":
    main()
