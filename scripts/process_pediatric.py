#!/usr/bin/env python3

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

import echonet


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("dest", type=click.Path(file_okay=False))
def main(src, dest):
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        with zf.open("Pediatric Echos/abnormals-deid/deid_coordinates.csv") as f:
            header = f.readline().decode("utf-8").strip().split(",")
            assert header == ['anon_patient_id', 'anon_accession_number', 'instance_number', 'frame_number', 'measurement', 'point_name', 'x', 'y']

            coordinates = collections.defaultdict(lambda: collections.defaultdict(list))
            frame = {}
            measurement = {}
            error = set()
            for line in f:
                patient, accession, instance, f, m, point_name, x, y = line.decode("utf-8").strip().split(",")

                instance = int(instance)
                f = int(f)
                point_name = int(point_name)
                x = int(x)
                y = int(y)

                key = (patient, accession, instance)

                coordinates[key][f].append((point_name, x, y))

                key = (patient, accession, instance, f)
                if key in measurement:
                    if measurement[key] != m:
                        measurement[key] = "INVALID"
                        error.add(key)
                measurement[key] = m

            for key in coordinates:
                for f in coordinates[key]:
                    coordinates[key][f] = np.array(sorted(coordinates[key][f]))[:, 1:]

        # os.makedirs(os.path.join(dest, "coordinates"), exist_ok=True)
        # for key in tqdm.tqdm(coordinates):
        #     for f in coordinates[key]:
        #         os.makedirs(os.path.join(dest, "coordinates", measurement[key + (f,)]), exist_ok=True)
        #         fig = plt.figure(figsize=(3, 3))
        #         plt.scatter(*coordinates[key][f].transpose(), s=1, color="k")
        #         for (i, (a, b)) in enumerate(coordinates[key][f]):
        #             plt.text(a, b, str(i))
        #         plt.tight_layout()
        #         plt.savefig(os.path.join(dest, "coordinates", measurement[key + (f,)], "_".join(map(str, key + (f,))) + ".pdf"))
        #         plt.close(fig)


        view = {}
        for (p, a, i, f) in measurement:
            m = measurement[(p, a, i, f)]
            a4c = ("A4C" in m)
            psax = ("psax" in m or "PSAX" in m)

            assert a4c != psax

            # TODO: check that only one view appears
            if a4c:
                view[(p, a, i)] = "A4C"
            elif psax:
                view[(p, a, i)] = "PSAX"

        instance_of_view = collections.defaultdict(lambda: collections.defaultdict(list))
        for (p, a, i) in view:
            instance_of_view[(p, a)][view[(p, a, i)]].append(i)
        # for (p, a, i) in view:
        #     if view[(p, a, i)] == "A4C":
        #             # print("{}-{}/*/*/

        ef = {}
        with zf.open("Pediatric Echos/abnormals-deid/deid_measurements.csv") as f:
            header = f.readline().decode("utf-8").strip().split(",")
            assert header == ['anon_patient_id', 'anon_accession_number', 'lv_ef_bullet', 'lv_ef_mod_a4c', 'lv_ef_mod_bp', 'lv_area_d_a4c', 'lv_area_s_a4c', 'lv_area_d_psax_pap', 'lv_area_s_psax_pap', 'lv_vol_d_bullet', 'lv_vol_s_bullet']

            for line in f:
                patient, accession, ef_bullet, *_ = line.decode("utf-8").strip().split(",")
                try:
                    ef_bullet = float(ef_bullet)
                    ef[(patient, accession)] = ef_bullet
                except:
                    print(ef_bullet)

        patients = sorted(set(p for (p, a) in ef))
        index = {p: i for (i, p) in enumerate(patients)}
        split = {}
        for p in index:
            split[p] = (index[p] % 10)
            # i = index[p] % 20
            # if i < 14:
            #     split[p] = "TRAIN"
            # elif i < 17:
            #     split[p] = "VAL"
            # else:
            #     split[p] = "TEST"
        with open(os.path.join(dest, "FileList.csv"), "w") as f:
            f.write("FileName,EF,Split\n")
            for (p, a) in ef:
                for i in instance_of_view[(p, a)]["A4C"]:
                    f.write("{}-{}-{:06d}.avi,{},{}\n".format(p, a, i, ef[(p, a)], split[p]))

        coord = {}
        iterables = [(zf, dest, coordinates, measurement, view, filename) for filename in zf.namelist()]
        with tqdm.tqdm(total=len(iterables)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                for c in executor.map(save_tgz, iterables):
                # for c in map(save_tgz, iterables):
                    # TODO: check keys of c not already in coord
                    coord.update(c)
                    pbar.update()

        for view in ["A4C", "PSAX"]:
            os.makedirs(os.path.join(dest, view), exist_ok=True)
            try:
                os.symlink(os.path.join("..", "Videos"), os.path.join(dest, view, "Videos"))
            except:
                pass
            with open(os.path.join(dest, view, "FileList.csv"), "w") as f:
                f.write("FileName,EF,Split\n")
                for (p, a) in ef:
                    for i in instance_of_view[(p, a)][view]:
                        if (p, a, "{:06d}".format(i)) in coord:
                            f.write("{}-{}-{:06d}.avi,{},{}\n".format(p, a, i, ef[(p, a)], split[p]))
            with open(os.path.join(dest, view, "VolumeTracings.csv"), "w") as f:
                f.write("FileName,X,Y,Frame\n")
                for (p, a) in ef:
                    for i in instance_of_view[(p, a)][view]:
                        if (p, a, "{:06d}".format(i)) in coord:
                            for (frame, c) in coord[p, a, "{:06d}".format(i)]:
                                for (x, y) in c:
                                    f.write("{}-{}-{:06d}.avi,{},{},{}\n".format(p, a, i, x, y, frame))


def save_tgz(x):
    (zf, dest, coordinates, measurement, view, filename) = x
    m = re.search("Pediatric Echos/abnormals-deid/dicom/(CR[0-9a-z]{7})-(CR[0-9a-z]{7}).tgz", filename)
    
    coord = collections.defaultdict(list)
    if m:
        patient, accession = m.groups()
        with zf.open(filename) as f:
            data = f.read()
    
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            for dicom in tf.getmembers():
                if dicom.isfile():
                    InstanceNumber, SOPInstanceUID = os.path.basename(dicom.name).split("-")
                    assert len(InstanceNumber) == 6
    
                    # output = os.path.join(dest, "videos", "{}-{}".format(patient, accession), os.path.splitext(dicom.name)[0] + ".avi")
                    # os.makedirs(os.path.dirname(output), exist_ok=True)
                    c = coordinates[patient, accession, int(InstanceNumber)]
                    output = {
                        "full": os.path.join(dest, "videos-full", "{}-{}-{}.avi".format(patient, accession, InstanceNumber)),
                        "crop": os.path.join(dest, "videos-crop", "{}-{}-{}.avi".format(patient, accession, InstanceNumber)),
                        "scale": os.path.join(dest, "Videos", "{}-{}-{}.avi".format(patient, accession, InstanceNumber)),
                    }
                    output.update({(f, "full"): os.path.join(dest, "trace-full", measurement[patient, accession, int(InstanceNumber), f], "{}-{}-{}-{}.jpg".format(patient, accession, InstanceNumber, f)) for f in c})
                    output.update({(f, "scale"): os.path.join(dest, "trace", measurement[patient, accession, int(InstanceNumber), f], "{}-{}-{}-{}.jpg".format(patient, accession, InstanceNumber, f)) for f in c})
                    for o in output.values():
                        os.makedirs(os.path.dirname(o), exist_ok=True)
    
                    if True:
                        with tf.extractfile(dicom.name) as f:
                            ds = pydicom.dcmread(f)
    
                        fps = 50
                        try:
                            fps = ds.CineRate  # TODO CineRate frequently missing
                        except:
                            pass
    
                        try:
                            video = ds.pixel_array.transpose((3, 0, 1, 2))
                            if view[patient, accession, int(InstanceNumber)] == "A4C":
                                video = video[:, :, ::-1, :]
                                for f in c:
                                    c[f][:, 1] = video.shape[2] - c[f][:, 1] - 1
                            else:
                                assert view[patient, accession, int(InstanceNumber)] == "PSAX"
                            video[1, :, :, :] = video[0, :, :, :]
                            video[2, :, :, :] = video[0, :, :, :]

                            if not os.path.isfile(output["full"]):
                                echonet.utils.savevideo(output["full"], video, fps=fps)
                            
                            for f in c:
                                frame = video[:, f, :, :].copy()
                                a, b = skimage.draw.polygon(c[f][:, 1], c[f][:, 0], (frame.shape[1], frame.shape[2]))
                                frame[2, a, b] = 255
                                PIL.Image.fromarray(frame.transpose((1, 2, 0))).save(output[f, "full"])

                            regions = ds.SequenceOfUltrasoundRegions
                            if len(regions) != 1:
                                raise ValueError("Found {} regions; expected 1.".format(len(regions)))
                            x0 = regions[0].RegionLocationMinX0
                            y0 = regions[0].RegionLocationMinY0
                            x1 = regions[0].RegionLocationMaxX1
                            y1 = regions[0].RegionLocationMaxY1
                            video = video[:, :, y0:(y1 + 1), x0:(x1 + 1)]
                            echonet.utils.savevideo(output["crop"], video, fps=fps)

                            _, _, h, w = video.shape
                            video = video[:, :, :, ((w - h) // 2):(h + (w - h) // 2)]
                            if video.shape[2] != video.shape[3]:
                                raise ValueError("Failed to make video square ({}, {})".format(video.shape[2], video.shape[3]))

                            for f in c:
                                c[f] -= np.array([x0 + ((w - h) // 2), y0])
                                c[f] = c[f] * 112 / np.array([video.shape[3], video.shape[2]])
                                c[f] = c[f].astype(np.int64)

                            video = np.array(list(map(lambda x: cv2.resize(x, (112, 112), interpolation=cv2.INTER_AREA), video.transpose((1, 2, 3, 0))))).transpose((3, 0, 1, 2))
                            echonet.utils.savevideo(output["scale"], video, fps=fps)

                            for f in c:
                                frame = video[:, f, :, :]
                                a, b = skimage.draw.polygon(c[f][:, 1], c[f][:, 0], (frame.shape[1], frame.shape[2]))
                                frame[2, a, b] = 255
                                PIL.Image.fromarray(frame.transpose((1, 2, 0))).save(output[f, "scale"])

                            coord[patient, accession, InstanceNumber].append((f, c[f]))
                        except Exception as e:
                            print(filename, dicom.name)
                            print(e, flush=True)
                            print("", flush=True)
    else:
        assert filename in ["Pediatric Echos/abnormals-deid/README.txt",
                            "Pediatric Echos/abnormals-deid/deid_measurements.csv",
                            "Pediatric Echos/abnormals-deid/deid_coordinates.csv"]
    return coord

if __name__ == "__main__":
    main()
