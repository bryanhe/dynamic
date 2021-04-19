#!/usr/bin/env python3

import math
import pandas
import torch
import torch.utils.data
import torchvision
import echonet
import os
import tqdm
import numpy as np
import scipy
import matplotlib.pyplot as plt
import concurrent.futures
import pickle

def main():
    output = "frame"
    block_size = 1024
    os.makedirs(os.path.join(output, "trace"), exist_ok=True)
    os.makedirs(os.path.join(output, "size"), exist_ok=True)
    os.makedirs(os.path.join(output, "lstm"), exist_ok=True)

    if False:
        save_segmentations(output, block_size)
    basename = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(os.path.join(output, "trace")))]

    try:
        with open(os.path.join(output, "size.pkl"), "rb") as f:
            size = pickle.load(f)
    except:
        size = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for s in tqdm.tqdm(executor.map(get_size, [os.path.join(output, "trace", b + ".npy") for b in  basename]), total=len(basename)):
                size.append(s)
        with open(os.path.join(output, "size.pkl"), "wb") as f:
            pickle.dump(size, f)
    # import collections
    # print(collections.Counter([s.shape[0] for s in size]))
    max_length = max([s.shape[0] for s in size])
    batch_size = 512
    num_workers = 0
    device = torch.device("cuda")

    dataset = echonet.datasets.Echo(split="all")
    systole_r = [dataset.frames[b][0] for b in basename]
    diastole_r = [dataset.frames[b][-1] for b in basename]
    systole_lookup = {b: dataset.frames[b][0] for b in basename}
    diastole_lookup = {b: dataset.frames[b][-1] for b in basename}
    with open(os.path.join(echonet.config.DATA_DIR, "FileList.csv")) as f:
        data = pandas.read_csv(f)
    data["FileName"] = data["FileName"].map(lambda x: os.path.splitext(x)[0])
    split = {split: set(data[data["Split"] == split]["FileName"]) for split in data["Split"].unique()}

    mask = [b in split["TRAIN"] for b in basename]
    size_train = np.concatenate([x for (x, m) in zip(size, mask) if m])
    mean = size_train.mean()
    std = size_train.std()
    dataset = {}
    for s in split:
        mask = [b in split[s] for b in basename]
        dataset[s] = Dataset(
            [x for (x, m) in zip(basename, mask) if m],
            [x for (x, m) in zip(size, mask) if m],
            [x for (x, m) in zip(systole_r, mask) if m],
            [x for (x, m) in zip(diastole_r, mask) if m],
            max_length=max_length,
            mean=mean,
            std=std)

    p = 1 / 50
    model = Sequence()
    model.linear.bias.data[:] = math.log(p)
    class LSTM(torch.nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.lstm = torch.nn.LSTM(input_size=1, hidden_size=64, num_layers=3, bidirectional=True)
            self.linear = torch.nn.Linear(64 * 2, 2)
        def forward(self, x):
            x = x.reshape(*x.shape, 1).transpose(0, 1)
            x = self.lstm(x)[0]
            x = self.linear(x)
            x = x.transpose(0, 1).transpose(1, 2)
            return x
    model = LSTM()
    model.linear.bias.data[:] = math.log(p)
    # breakpoint()
    model.to(device)
    try:
        # raise ValueError()
        model.load_state_dict(torch.load(os.path.join(output, "model.pt")))
    except:
        optim = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, 50)
        for epoch in range(100):
            for s in ["TRAIN", "VAL"]:
                model.train(s == "TRAIN")
                print("Epoch #{} {}".format(epoch, s))
                dataloader = torch.utils.data.DataLoader(dataset[s], batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
                total = 0.
                n = 0
                with tqdm.tqdm(total=len(dataloader)) as pbar:
                    for (_, x, y) in dataloader:
                        x = x.to(device)
                        y = y.to(device)
                        mask = ~torch.logical_or(torch.isnan(torch.stack((x, x), dim=1)), torch.isnan(y))
                        x[torch.isnan(x)] = -10
                        yhat = model(x)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(yhat[mask], y[mask], reduction="sum")
                        if s == "TRAIN":
                            optim.zero_grad()
                            loss.backward()
                            optim.step()

                        # torch.nn.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
                        total += loss.item()
                        n += mask.sum().item()

                        pbar.set_description("Loss: {:.4f}".format(total / n))
                        pbar.update()
            scheduler.step()
        torch.save(model.state_dict(), os.path.join(output, "model.pt"))

    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset["TEST"], batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), drop_last=False)
    total = 0.
    n = 0
    systole_p = []
    diastole_p = []
    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (filename, x, y) in dataloader:
                x = x.to(device)
                y = y.to(device)
                valid = ~torch.isnan(x)
                mask = ~torch.logical_or(torch.isnan(torch.stack((x, x), dim=1)), torch.isnan(y))
                x[torch.isnan(x)] = -10
                yhat = model(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(yhat[mask], y[mask], reduction="sum")
                prob = torch.sigmoid(yhat)

                for (f, s, p, v) in zip(filename, x.cpu(), prob.cpu(), valid):
                    systole_p.append(scipy.signal.find_peaks([-math.inf] + list(p[0, v]) + [-math.inf], height=0.05, distance=5)[0] - 1)
                    diastole_p.append(scipy.signal.find_peaks([-math.inf] + list(p[1, v]) + [-math.inf], height=0.05, distance=5)[0] - 1)

                    if f in [
                        "0X17828CD670289D36",
                        "0X179B579A585FF160",
                        "0X18BA5512BE5D6FFA",
                        "0X19E42A10F0077B9F",
                        "0X1A3D565B371DC573",
                        "0X1B2BCDAE290F6015",
                        "0X1CDE7FECA3A1754B",
                        "0X1CF4B07994B62DBB",
                        "0X1D865EBAD1A947E4",
                        "0X1E50146E7EAFF53E",
                        "0X1EB9E86F4FA26B5B",
                        "0X2012F90A3894AE6C",
                        "0X20F3D05837705ED2",
                        "0X211D307253ACBEE7",
                        "0X233DCAFBF90253C7",
                        # ]:
                        "0X18BA5512BE5D6FFA",
                        "0X19359F9246BB8A33",
                        "0X19E42A10F0077B9F",
                        "0X1A3D565B371DC573",
                        "0X1B2BCDAE290F6015",
                        "0X1C8C0CE25970C40",
                        "0X1D181F5019010E4B",
                        "0X1E50146E7EAFF53E",
                        "0X211D307253ACBEE7",
                        "0X233DCAFBF90253C7",
                        "0X26444E1ACD4FE90F",
                        "0X272A1770B9F8E6A0",
                        "0X28712788DD9BC1B6",
                        "0X287AFCC7F6DEED83",
                        "0X2B9185D91BBE0E97",
                        "0X2BF66637343A27ED",
                        "0X2D3E3C182D1459C8",
                        "0X30DF42C999969D67",
                        "0X6D1D29802905D6E0",
                        ]:

                        fig, ax = plt.subplots(3, figsize=(8, 6))

                        ax[0].plot(std * s[v] + mean)
                        ylim = ax[0].get_ylim()
                        ax[0].plot([systole_lookup[f]] * 2, ylim, linewidth=1, color="k")
                        ax[0].plot([diastole_lookup[f]] * 2, ylim, linewidth=1, color="k")
                        ax[0].set_ylim(ylim)

                        ax[1].plot(p[0, v])
                        ylim = ax[1].get_ylim()
                        for x in systole_p[-1]:
                            ax[1].plot([x, x], ylim, linewidth=1, color="k")
                        ax[1].set_ylim(ylim)

                        ax[2].plot(p[1, v])
                        ylim = ax[1].get_ylim()
                        for x in diastole_p[-1]:
                            ax[2].plot([x, x], ylim, linewidth=1, color="k")
                        ax[2].set_ylim(ylim)

                        plt.tight_layout()
                        plt.savefig(os.path.join(output, "lstm", f + ".pdf"))
                        plt.close(fig)
                        print(f)

                # torch.nn.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
                total += loss.item()
                n += mask.sum().item()

                pbar.set_description("Loss: {:.3f}".format(total / n))
                pbar.update()

    ans = []
    mask = [b in split["TEST"] for b in basename]
    for (filename, pred, real) in zip([x for (x, m) in zip(basename, mask) if m], systole_p, [x for (x, m) in zip(systole_r, mask) if m]):
        errors = [p - real for p in pred]
        close = [e for e in errors if abs(e) <= 25]
        if len(close) == 1:
            ans.append(abs(close[0]))
        else:
            ans.append(25)
    print("Systole:", sum(ans) / len(ans))

    ans = []
    mask = [b in split["TEST"] for b in basename]
    for (filename, pred, real) in zip([x for (x, m) in zip(basename, mask) if m], diastole_p, [x for (x, m) in zip(diastole_r, mask) if m]):
        errors = [p - real for p in pred]
        close = [e for e in errors if abs(e) <= 25]
        if len(close) == 1:
            ans.append(abs(close[0]))
        else:
            # print(filename, len(close))
            ans.append(25)
    print("Diastole:", sum(ans) / len(ans))


    # src_des = [(os.path.join(output, "trace", filename), os.path.join(output, "size", os.path.splitext(filename)[0] + ".pdf")) for filename in os.listdir(os.path.join(output, "trace"))]
    systole_p = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for s in tqdm.tqdm(executor.map(heuristic, size), total=len(size)):
        # for filename in tqdm.tqdm(map(heuristic, src_des), total=len(src_des)):
            systole_p.append(s)

    ans = []
    ans = []
    for (filename, pred, real, s) in zip(basename, systole_p, systole_r, size):
        errors = [p - real for p in pred]
        close = [e for e in errors if abs(e) <= 25]
        if len(close) == 1:
            ans.append(close[0])
            if abs(close[0]) > 15:
                # print(filename, close[0])
                plot(s, filename, output, real)
        # else:
        #     ans.append(25)
    offset = sum(ans) / len(ans)
    fig = plt.figure(figsize=(3, 3))
    plt.hist(ans, bins=range(-25, 26))
    plt.tight_layout()
    plt.savefig("asd.pdf")
    plt.close(fig)
    print("offset", offset)

    ans = []
    ans_offset = []
    for (filename, pred, real, s) in zip(basename, systole_p, systole_r, size):
        errors = [p - real for p in pred]
        close = [e for e in errors if abs(e) <= 25]
        # print(len(close))
        if len(close) == 1:
            ans.append(abs(close[0]))
            ans_offset.append(abs(close[0] - offset))
        else:
            ans.append(25)
            ans_offset.append(25)
    print("Heuristic:", sum(ans) / len(ans))
    print("Heuristic (offset):", sum(ans_offset) / len(ans_offset))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, size, systole, diastole, max_length, mean=0, std=1, tol=25):
        self.filename = filename
        self.size = size
        self.systole = systole
        self.diastole = diastole
        self.max_length = max_length
        self.mean = mean
        self.std = std
        self.tol = tol

    def __getitem__(self, index):
        x = self.size[index]
        x = x.astype(np.float32)
        x -= self.mean
        x /= self.std
        x = np.concatenate((x, np.full(self.max_length - x.shape[0], math.nan, dtype=np.float32)))

        y = np.full((2, x.shape[0]), math.nan)
        y[0, max(0, self.systole[index] - 25):self.systole[index] + 25] = 0
        y[0, self.systole[index]] = 1
        y[1, max(0, self.diastole[index] - 25):self.diastole[index] + 25] = 0
        y[1, self.diastole[index]] = 1

        return self.filename[index], x, y

    def __len__(self):
        return len(self.size)

class Sequence(torch.nn.Module):
    # Based on https://github.com/pytorch/examples/blob/master/time_sequence_prediction/train.py
    def __init__(self, dim=64):
        super(Sequence, self).__init__()
        self.dim = dim
        self.lstm1 = torch.nn.LSTMCell(1, dim)
        self.lstm2 = torch.nn.LSTMCell(dim, dim)
        self.linear = torch.nn.Linear(dim, 2)
        self.h_t = torch.nn.Parameter(torch.zeros(1, self.dim))
        self.c_t = torch.nn.Parameter(torch.zeros(1, self.dim))
        self.h_t2 = torch.nn.Parameter(torch.zeros(1, self.dim))
        self.c_t2 = torch.nn.Parameter(torch.zeros(1, self.dim))

    def forward(self, input):
        outputs = []
        h_t = self.h_t.repeat(input.size(0), 1)
        c_t = self.c_t.repeat(input.size(0), 1)
        h_t2 = self.h_t2.repeat(input.size(0), 1)
        c_t2 = self.c_t2.repeat(input.size(0), 1)

        for (i, input_t) in enumerate(input.split(1, dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, dim=2)
        return outputs


def save_segmentations(output, block_size):
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


def get_size(filename):
    logit = np.load(filename)
    size = (logit > 0).sum((1, 2))
    return size

def heuristic(size):
    # Identify systole frames with peak detection
    trim_min = sorted(size)[round(len(size) ** 0.05)]
    trim_max = sorted(size)[round(len(size) ** 0.95)]
    trim_range = trim_max - trim_min
    systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])
    return systole

def plot(size, filename, output, systole):
    # Plot sizes
    fig = plt.figure(num=filename, figsize=(size.shape[0] / 50 * 5, 3))
    plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
    ylim = plt.ylim()
    s = systole
    plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
    systole = heuristic(size)
    for s in systole:
        plt.plot(np.array([s, s]) / 50, ylim, "--", linewidth=1)
    plt.ylim(ylim)
    plt.title(filename)
    plt.xlabel("Seconds")
    plt.ylabel("Size (pixels)")
    plt.tight_layout()
    plt.savefig(os.path.join(output, "size", filename + ".pdf"))
    plt.close(fig)

if __name__ == "__main__":
    main()
