"""Functions for training and running segmentation."""

import math
import sklearn
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm

import echonet


def run(num_epochs=50,
        modelname="deeplabv3_resnet50",
        pretrained=False,
        output=None,
        device=None,
        n_train_patients=None,
        num_workers=8,
        batch_size=8,
        seed=0,
        lr_step_period=None,
        save_segmentation=False,
        block_size=1024,
        run_test=False):
    """Trains/tests segmentation model.

    Args:
        num_epochs (int, optional): Number of epochs during training
            Defaults to 50.
        modelname (str, optional): Name of segmentation model. One of ``deeplabv3_resnet50'',
            ``deeplabv3_resnet101'', ``fcn_resnet50'', or ``fcn_resnet101''
            (options are torchvision.models.segmentation.<modelname>)
            Defaults to ``deeplabv3_resnet50''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to False.
        output (str or None, optional): Name of directory to place outputs
            Defaults to None (replaced by output/segmentation/<modelname>_<pretrained/random>/).
        device (str or None, optional): Name of device to run on. See
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            for options. If ``None'', defaults to ``cuda'' if available, and ``cpu'' otherwise.
            Defaults to ``None''.
        n_train_patients (str or None, optional): Number of training patients. Used to ablations
            on number of training patients. If ``None'', all patients used.
            Defaults to ``None''.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 20.
        seed (int, optional): Seed for random number generator.
            Defaults to 0.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            If ``None'', learning rate is not decayed.
            Defaults to ``None''.
        save_segmentation (bool, optional): Whether to save videos with segmentations.
            Defaults to False.
        block_size (int, optional): Number of frames to segment simultaneously when saving
            videos with segmentation (this is used to adjust the memory usage on GPU; decrease
            this is GPU memory issues occur).
            Defaults to 1024.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
    """

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "segmentation", "{}_{}".format(modelname, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    # model = torchvision.models.segmentation.__dict__[modelname](pretrained=pretrained, aux_loss=False)
    model = echonet.models.r3d_18()

    p1 = 0.09
    p2 = 1 / 112 / 112
    # model.classifier = torch.nn.Conv3d(model.classifier.in_channels, 3, kernel_size=model.classifier.kernel_size)  # change number of outputs to 1
    model.classifier[-1] = torch.nn.Conv3d(model.classifier[-1].in_channels, 3, kernel_size=model.classifier[-1].kernel_size)  # change number of outputs to 1
    # model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 3, kernel_size=model.classifier[-1].kernel_size)  # change number of outputs to 1
    w = [math.log(p1), math.log(p2), math.log(p2)]
    model.classifier[-1].weight.data[:] = 0
    model.classifier[-1].bias.data = torch.as_tensor(w)
    # model.classifier.weight.data[:] = 0
    # model.classifier.bias.data = torch.as_tensor(w)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    # op_ft = torch.optim.SGD(model.module.classifier[-1].parameters(), lr=1e-6, momentum=0.9)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Compute mean and std
    # tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace", "LargeApex", "SmallApex", "LargeBase", "SmallBase"]
    # dataset = echonet.datasets.Echo(split="train", target_type=tasks)
    # os.makedirs("trace/large", exist_ok=True)
    # os.makedirs("trace/small", exist_ok=True)
    # os.makedirs("trace/test", exist_ok=True)
    # for i in range(10):
    #     (_, (large_frame, small_frame, large_trace, small_trace, large_apex, small_apex, large_base, small_base)) = dataset[i]
    #     import PIL
    #     x = large_frame.astype(np.uint8)
    #     x = x.transpose((1, 2, 0))
    #     x[large_trace > 0, 2] = 255
    #     x[scipy.ndimage.binary_dilation(large_apex), 0] = 255
    #     x[scipy.ndimage.binary_dilation(large_base), 1] = 255
    #     # x[large_apex > 0, 0] = 255
    #     # x[small_apex > 0, 1] = 255
    #     PIL.Image.fromarray(x).save("trace/large/img_{:06d}.tif".format(i))
    # breakpoint()
    # for basename in sorted(dataset.trace.keys()):
    #     t = dataset.trace[basename]
    #     t = t[sorted(t.keys())[0]]
    #     fig = plt.figure(figsize=(9, 9))
    #     for (i, (x1, y1, x2, y2)) in enumerate(t):
    #         plt.text(x1, y1, str(i))
    #         plt.plot([x1, x2], [y1, y2])
    #     
    #     
    #     plt.tight_layout()
    #     plt.savefig("trace/test/{}.pdf".format(basename))
    #     plt.close(fig)

    # breakpoint()

    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"), num_workers=num_workers)
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace", "LargeApex", "SmallApex", "LargeBase", "SmallBase"]
    kwargs = {
        "target_type": tasks,
        "mean": mean,
        "std": std
    }

    # Set up datasets and dataloaders
    train_dataset = echonet.datasets.Echo(split="train", **kwargs)

    if n_train_patients is not None and len(train_dataset) > n_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(len(train_dataset), n_train_patients, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        echonet.datasets.Echo(split="val", **kwargs), batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # Run training and testing loops
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

                if False: # epoch == 0:
                    loss, large_inter, large_union, small_inter, small_union = echonet.utils.segmentation.run_epoch(model, dataloaders[phase], phase == "train", op_ft, device)
                else:
                    loss, large_inter, large_union, small_inter, small_union = echonet.utils.segmentation.run_epoch(model, dataloaders[phase], phase == "train", optim, device)
                overall_dice = 2 * (large_inter.sum() + small_inter.sum()) / (large_union.sum() + large_inter.sum() + small_union.sum() + small_inter.sum())
                large_dice = 2 * large_inter.sum() / (large_union.sum() + large_inter.sum())
                small_dice = 2 * small_inter.sum() / (small_union.sum() + small_inter.sum())
                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                    phase,
                                                                    loss,
                                                                    overall_dice,
                                                                    large_dice,
                                                                    small_dice,
                                                                    time.time() - start_time,
                                                                    large_inter.size,
                                                                    sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                    sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count())),
                                                                    batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        # Load best weights
        checkpoint = torch.load(os.path.join(output, "best.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if run_test:
            # Run on validation and test
            # for split in ["val", "test"]:
            for split in ["val", "test"]:
                dataset = echonet.datasets.Echo(split=split, **kwargs)
                dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                loss, large_inter, large_union, small_inter, small_union = echonet.utils.segmentation.run_epoch(model, dataloader, False, None, device)

                overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
                large_dice = 2 * large_inter / (large_union + large_inter)
                small_dice = 2 * small_inter / (small_union + small_inter)
                with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                    g.write("Filename, Overall, Large, Small\n")
                    for (filename, overall, large, small) in zip(dataset.fnames, overall_dice, large_dice, small_dice):
                        g.write("{},{},{},{}\n".format(filename, overall, large, small))

                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)), echonet.utils.dice_similarity_coefficient)))
                f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(large_inter, large_union, echonet.utils.dice_similarity_coefficient)))
                f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(small_inter, small_union, echonet.utils.dice_similarity_coefficient)))
                f.flush()


    tasks = ["Filename", "EF", "LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace", "LargeApex", "SmallApex", "LargeBase", "SmallBase"]
    kwargs = {
        "target_type": tasks,
        "mean": mean,
        "std": std
    }
    dataset = echonet.datasets.Echo(split="test", **kwargs)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))


    model.eval()
    ef_real = []
    ef_pred = []
    os.makedirs(os.path.join(output, "disk"), exist_ok=True)
    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (_, (filename, ef, large_frame, small_frame, large_trace, small_trace, large_apex, small_apex, large_base, small_base)) in dataloader:
                ef_real.extend(ef.numpy())
                # Run prediction for diastolic frames and compute loss
                large_frame = large_frame.to(device)
                yhat = model(large_frame)["out"]
                # trace = torch.sigmoid(yhat[:, 0, :, :])
                # apex = torch.sigmoid(yhat[:, 1, :, :])
                # base = torch.sigmoid(yhat[:, 2, :, :])
                trace = yhat[:, 0, :, :]
                apex = yhat[:, 1, :, :]
                base = yhat[:, 2, :, :]
                edv = []
                for (fn, t) in zip(filename, trace.cpu().numpy()):
                    os.makedirs(os.path.join(output, "disk", os.path.splitext(fn)[0]), exist_ok=True)
                    v, *_ = echonet.utils.volume.calculateVolumeMainAxisTopShift(t, 20, pointShifts=1, output=os.path.join(output, "disk", os.path.splitext(fn)[0], "diastole_computer"))
                    assert len(v.values()) == 1
                    edv.append(list(v.values())[0])
                for (fn, t) in zip(filename, large_trace.cpu().numpy()):
                    v, *_ = echonet.utils.volume.calculateVolumeMainAxisTopShift(t, 20, pointShifts=1, output=os.path.join(output, "disk", os.path.splitext(fn)[0], "diastole_human"))
                    assert len(v.values()) == 1
                    # edv.append(list(v.values())[0])


                # edv = ((trace > 0).sum(2) ** 2).sum(1)

                small_frame = small_frame.to(device)
                yhat = model(small_frame)["out"]
                # trace = torch.sigmoid(yhat[:, 0, :, :])
                # apex = torch.sigmoid(yhat[:, 1, :, :])
                # base = torch.sigmoid(yhat[:, 2, :, :])
                trace = yhat[:, 0, :, :]
                apex = yhat[:, 1, :, :]
                base = yhat[:, 2, :, :]
                # trace = trace.cpu().numpy()
                # trace = small_trace.cpu().numpy()
                esv = []
                for (fn, t) in zip(filename, trace.cpu().numpy()):
                    v, *_ = echonet.utils.volume.calculateVolumeMainAxisTopShift(t, 20, pointShifts=1, output=os.path.join(output, "disk", os.path.splitext(fn)[0], "systole_computer"))
                    assert len(v.values()) == 1
                    esv.append(list(v.values())[0])
                for (fn, t) in zip(filename, small_trace.cpu().numpy()):
                    v, *_ = echonet.utils.volume.calculateVolumeMainAxisTopShift(t, 20, pointShifts=1, output=os.path.join(output, "disk", os.path.splitext(fn)[0], "systole_human"))
                    assert len(v.values()) == 1
                    # esv.append(list(v.values())[0])
                # esv = ((trace > 0).sum(2) ** 2).sum(1)

                edv = np.array(edv)
                esv = np.array(esv)
                ef_pred.extend((100 * (1 - esv / edv)))

                for (fn, ef) in zip(filename, 1 - esv / edv):
                    if ef < 0:
                        print(fn)

                # for (p, fn) in zip(

                print(sklearn.metrics.r2_score(ef_real, ef_pred))
                pbar.update()
    fig = plt.figure(figsize=(3, 3))
    plt.scatter(ef_real, ef_pred, s=1, color="k")
    plt.xlabel("Real")
    plt.ylabel("Prediction")
    plt.axis([0, 100, 0, 100])
    plt.tight_layout()
    plt.savefig("seg_ef_prediction.pdf")
    plt.close(fig)
    mask = [0 < e < 100 for e in ef_pred]
    mask = [abs(r - p) < 10 for (r, p) in zip(ef_real, ef_pred)]
    print(sklearn.metrics.r2_score([e for (e, m) in zip(ef_real, mask) if m], [e for (e, m) in zip(ef_pred, mask) if m]))
    print(scipy.stats.linregress([e for (e, m) in zip(ef_real, mask) if m], [e for (e, m) in zip(ef_pred, mask) if m]))



    # Saving videos with segmentations
    dataset = echonet.datasets.Echo(split="test",
                                    target_type=["Filename", "LargeIndex", "SmallIndex"],  # Need filename for saving, and human-selected frames to annotate
                                    mean=mean, std=std,  # Normalization
                                    length=None, max_length=None, period=1  # Take all frames
                                    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=num_workers, shuffle=False, pin_memory=False, collate_fn=_video_collate_fn)

    # Save videos with segmentation
    if save_segmentation and not all(os.path.isfile(os.path.join(output, "videos", f)) for f in dataloader.dataset.fnames):
        # TODO: move to separate function
        # TODO: don't do a binary all-done check (if half of files are done, just run on that half)
        # Only run if missing videos

        model.eval()

        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output, "size"), exist_ok=True)
        echonet.utils.latexify()

        with torch.no_grad():
            with open(os.path.join(output, "size.csv"), "w") as g:
                g.write("Filename,Frame,Size,HumanLarge,HumanSmall,ComputerSmall\n")
                for (x, (filenames, large_index, small_index), length) in tqdm.tqdm(dataloader):
                    # Run segmentation model on blocks of frames one-by-one
                    # The whole concatenated video may be too long to run together
                    y = np.concatenate([model(x[i:(i + block_size), :, :, :].to(device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], block_size)])

                    start = 0
                    x = x.numpy()
                    for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                        print(filename)
                        # Extract one video and segmentation predictions
                        video = x[start:(start + offset), ...]
                        pred = y[start:(start + offset), :, :, :]
                        logit = pred[:, 0, :, :]
                        apex = pred[:, 1, :, :]
                        base = pred[:, 2, :, :]

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

                        apex = 1 / (1 + np.exp(-apex))
                        apex /= apex.max((1, 2)).reshape((-1, 1, 1))
                        video[:, 1, :, w:] = np.maximum(255. * apex, video[:, 1, :, w:])  # pylint: disable=E1111

                        base = 1 / (1 + np.exp(-base))
                        base /= base.max((1, 2)).reshape((-1, 1, 1))
                        video[:, 2, :, w:] = np.maximum(255. * base, video[:, 2, :, w:])  # pylint: disable=E1111

                        # Add blank canvas under pair of videos
                        video = np.concatenate((video, np.zeros_like(video)), 2)

                        # Compute size of segmentation per frame
                        size = (logit > 0).sum((1, 2))

                        # Identify systole frames with peak detection
                        trim_min = sorted(size)[round(len(size) ** 0.05)]
                        trim_max = sorted(size)[round(len(size) ** 0.95)]
                        trim_range = trim_max - trim_min
                        systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])

                        # Write sizes and frames to file
                        for (frame, s) in enumerate(size):
                            g.write("{},{},{},{},{},{}\n".format(filename, frame, s, 1 if frame == large_index[i] else 0, 1 if frame == small_index[i] else 0, 1 if frame in systole else 0))

                        # Plot sizes
                        fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                        plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                        ylim = plt.ylim()
                        for s in systole:
                            plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                        plt.ylim(ylim)
                        plt.title(os.path.splitext(filename)[0])
                        plt.xlabel("Seconds")
                        plt.ylabel("Size (pixels)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output, "size", os.path.splitext(filename)[0] + ".pdf"))
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

                            if f == large_index[i]:
                                # If frame is human-selected diastole, mark with green dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 225, 0]).reshape((1, 3, 1))
                            if f == small_index[i]:
                                # If frame is human-selected systole, mark with red dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 0, 225]).reshape((1, 3, 1))

                            # Get pixels for a circle centered on the pixel
                            r, c = skimage.draw.circle(int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10)), 4.1)

                            # On the frame that's being shown, put a circle over the pixel
                            video[f, :, r, c] = 255.

                        # Rearrange dimensions and save
                        video = video.transpose(1, 0, 2, 3)
                        video = video.astype(np.uint8)
                        echonet.utils.savevideo(os.path.join(output, "videos", filename), video, 50)

                        # Move to next video
                        start += offset


def run_epoch(model, dataloader, train, optim, device):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """

    total = 0.
    total2 = 0
    total3 = 0
    n = 0

    pos = 0
    neg = 0
    pos_pix = 0
    neg_pix = 0

    model.train(train)

    large_inter = 0
    large_union = 0
    small_inter = 0
    small_union = 0
    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (_, (large_frame, small_frame, large_trace, small_trace, large_apex, small_apex, large_base, small_base)) in dataloader:
                # Count number of pixels in/out of human segmentation
                large_mask = ~torch.isnan(large_trace).any(3).any(2)
                small_mask = ~torch.isnan(small_trace).any(3).any(2)
                pos += (large_trace[large_mask] == 1).sum().item()
                pos += (small_trace[small_mask] == 1).sum().item()
                neg += (large_trace[large_mask] == 0).sum().item()
                neg += (small_trace[small_mask] == 0).sum().item()

                # Count number of pixels in/out of computer segmentation
                pos_pix += (large_trace[large_mask] == 1).sum(0).numpy()
                pos_pix += (small_trace[small_mask] == 1).sum(0).numpy()
                neg_pix += (large_trace[large_mask] == 0).sum(0).numpy()
                neg_pix += (small_trace[small_mask] == 0).sum(0).numpy()

                # Run prediction for diastolic frames and compute loss
                large_frame = large_frame.to(device)
                target = torch.stack((large_trace, large_apex, large_base), dim=1)
                target = target.transpose(1, 2)[large_mask]
                target = target.to(device)
                y_large = model(large_frame)["out"]
                y_large = y_large.transpose(1, 2)[large_mask]
                # loss_large = torch.nn.functional.binary_cross_entropy_with_logits(y_large, target, reduction="sum")
                l = torch.nn.functional.binary_cross_entropy_with_logits(y_large, target, reduction="none")
                l = l.sum((0, 2, 3))
                l[1:] *= 100
                loss_large = l
                large_trace = large_trace[large_mask]
                # Compute pixel intersection and union between human and computer segmentations
                large_inter += np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_union += np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_inter_list.extend(np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                large_union_list.extend(np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                # Run prediction for systolic frames and compute loss
                small_frame = small_frame.to(device)
                target = torch.stack((small_trace, small_apex, small_base), dim=1)
                target = target.transpose(1, 2)[small_mask]
                target = target.to(device)
                y_small = model(small_frame)["out"]
                y_small = y_small.transpose(1, 2)[small_mask]
                l = torch.nn.functional.binary_cross_entropy_with_logits(y_small, target, reduction="none")
                l = l.sum((0, 2, 3))
                l[1:] *= 100
                loss_small = l
                small_trace = small_trace[small_mask]
                # Compute pixel intersection and union between human and computer segmentations
                small_inter += np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_union += np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_inter_list.extend(np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                small_union_list.extend(np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                # Take gradient step if training
                loss = (loss_large + loss_small) / 2
                if train:
                    optim.zero_grad()
                    loss.sum().backward()
                    optim.step()

                # Accumulate losses and compute baselines
                total += loss[0].item()
                total2 += loss[1].item()
                total3 += loss[2].item()
                n += large_trace.size(0)
                p = pos / (pos + neg)
                p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)

                # Show info on process bar
                pbar.set_postfix_str("{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(total / n / 112 / 112, loss[0].item() / large_trace.size(0) / 112 / 112, -p * math.log(p) - (1 - p) * math.log(1 - p), (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(), total2 / n / 112 / 112, total3 / n / 112 / 112, 2 * large_inter / (large_union + large_inter), 2 * small_inter / (small_union + small_inter)))
                pbar.update()

    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)

    return (total / n / 112 / 112,
            large_inter_list,
            large_union_list,
            small_inter_list,
            small_union_list,
            )


def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos.

    This function should be used in a dataloader for a dataset that returns
    a video as the first element, along with some (non-zero) tuple of
    targets. Then, the input x is a list of tuples:
      - x[i][0] is the i-th video in the batch
      - x[i][1] are the targets for the i-th video

    This function returns a 3-tuple:
      - The first element is the videos concatenated along the frames
        dimension. This is done so that videos of different lengths can be
        processed together (tensors cannot be "jagged", so we cannot have
        a dimension for video, and another for frames).
      - The second element is contains the targets with no modification.
      - The third element is a list of the lengths of the videos in frames.
    """
    video, target = zip(*x)  # Extract the videos and targets

    # ``video'' is a tuple of length ``batch_size''
    #   Each element has shape (channels=3, frames, height, width)
    #   height and width are expected to be the same across videos, but
    #   frames can be different.

    # ``target'' is also a tuple of length ``batch_size''
    # Each element is a tuple of the targets for the item.

    i = list(map(lambda t: t.shape[1], video))  # Extract lengths of videos in frames

    # This contatenates the videos along the the frames dimension (basically
    # playing the videos one after another). The frames dimension is then
    # moved to be first.
    # Resulting shape is (total frames, channels=3, height, width)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))

    # Swap dimensions (approximately a transpose)
    # Before: target[i][j] is the j-th target of element i
    # After:  target[i][j] is the i-th target of element j
    target = zip(*target)

    return video, target, i
