#!/usr/bin/env python3

import time
import math
import sklearn
import torch
import torchvision
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
import matplotlib.pyplot as plt

import echonet


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False))
@click.argument("dest", type=click.Path(file_okay=False))
def main(src, dest, device=None):
    num_epochs = 10
    weight_decay=1e-4
    lr_step_period=15
    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelname = "r2plus1d_18"
    pretrained = False
    # Set up model
    model = torchvision.models.video.__dict__[modelname](pretrained=pretrained)

    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    checkpoint = torch.load("r2plus1d_18_32_2_pretrained.pt")  # TODO
    model.load_state_dict(checkpoint['state_dict'])
    #         optim.load_state_dict(checkpoint['opt_dict'])
    #         scheduler.load_state_dict(checkpoint['scheduler_dict'])
    #         epoch_resume = checkpoint["epoch"] + 1
    #         bestLoss = checkpoint["best_loss"]
    #         f.write("Resuming from epoch {}\n".format(epoch_resume))



    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=src, split="train"))
    tasks = "EF"
    frames = 32
    period = 2
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }

    batch_size = 20
    num_workers = 5
    split="test"
    dataloader = torch.utils.data.DataLoader(
        echonet.datasets.Echo(root=src, split="test", **kwargs),
        batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
    loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device)
    print("{} blind (one clip) R2:   {:.3f} ({:.3f} - {:.3f})".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
    print("{} blind (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
    print("{} blind (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))

    os.makedirs(os.path.join(dest, "fig"), exist_ok=True)
    fig = plt.figure(figsize=(3, 3))
    plt.scatter(y, yhat, s=1, color="k")
    plt.savefig(os.path.join(dest, "fig", "scatter_blind.pdf"))
    plt.close(fig)

    optim = torch.optim.SGD(model.module.fc.parameters(), lr=1e-5, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }
    print("C", flush=True)

    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = echonet.datasets.Echo(root=src, split="train", **kwargs, pad=12, rotate=None)
    # if n_train_patients is not None and len(dataset["train"]) > n_train_patients:
    #     # Subsample patients (used for ablation experiment)
    #     indices = np.random.choice(len(dataset["train"]), n_train_patients, replace=False)
    #     dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = echonet.datasets.Echo(root=src, split="val", **kwargs)

    output = os.path.join(dest, "transfer")
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]

                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, phase == "train", optim, device)
                f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                              phase,
                                                              loss,
                                                              sklearn.metrics.r2_score(y, yhat),
                                                              time.time() - start_time,
                                                              y.size,
                                                              sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                              sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count())),
                                                              batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': loss,
                'r2': sklearn.metrics.r2_score(y, yhat),
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss



    split="test"
    ds = echonet.datasets.Echo(root=src, split="test", **kwargs)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
    loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device)
    print("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
    print("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
    print("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))

    os.makedirs(os.path.join(dest, "fig"), exist_ok=True)
    fig = plt.figure(figsize=(3, 3))
    plt.scatter(y, yhat, s=1, color="k")
    plt.savefig(os.path.join(dest, "fig", "scatter_transfer.pdf"))
    plt.close(fig)

    optim = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }
    print("C", flush=True)

    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = echonet.datasets.Echo(root=src, split="train", **kwargs, pad=12, rotate=None)
    # if n_train_patients is not None and len(dataset["train"]) > n_train_patients:
    #     # Subsample patients (used for ablation experiment)
    #     indices = np.random.choice(len(dataset["train"]), n_train_patients, replace=False)
    #     dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = echonet.datasets.Echo(root=src, split="val", **kwargs)

    output = os.path.join(dest, "full")
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]

                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, phase == "train", optim, device)
                f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                              phase,
                                                              loss,
                                                              sklearn.metrics.r2_score(y, yhat),
                                                              time.time() - start_time,
                                                              y.size,
                                                              sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                              sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count())),
                                                              batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': loss,
                'r2': sklearn.metrics.r2_score(y, yhat),
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss



    split="test"
    ds = echonet.datasets.Echo(root=src, split="test", **kwargs)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
    loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device)
    print("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
    print("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
    print("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))

    os.makedirs(os.path.join(dest, "fig"), exist_ok=True)
    fig = plt.figure(figsize=(3, 3))
    plt.scatter(y, yhat, s=1, color="k")
    plt.savefig(os.path.join(dest, "fig", "scatter_ft.pdf"))
    plt.close(fig)




if __name__ == "__main__":
    main()
