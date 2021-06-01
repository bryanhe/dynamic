#!/usr/bin/env python3

import math
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

import echonet


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False))
@click.argument("dest", type=click.Path(file_okay=False))
def main(src, dest):
    echonet.utils.latexify()

    BATCHES = ["abnormals-deid", "Second Batch"]
    os.makedirs(dest, exist_ok=True)

    coordinates = collections.defaultdict(lambda: collections.defaultdict(list))  # coordinates[patient, accession, instance][frame] = ???
    frame = {}
    measurement = {}
    error = set()

    view = {}
    instance_of_view = collections.defaultdict(lambda: collections.defaultdict(list))

    ef = {}
    sex = {}
    age = {}
    weight = {}
    height = {}

    for batch in BATCHES:
        ### Read trace coordinates ###
        with open(os.path.join(src, batch, "deid_coordinates.csv")) as f:
            header = f.readline().strip().split(",")
            assert header == ['anon_patient_id', 'anon_accession_number', 'instance_number', 'frame_number', 'measurement', 'point_name', 'x', 'y']

            for line in f:
                patient, accession, instance, f, m, point_name, x, y = line.strip().split(",")

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
                        # TODO: a few frames have a4c and psax traces; just manually filter?
                        measurement[key] = "INVALID"
                        error.add(key)
                else:
                    measurement[key] = m

        ### Read metadata ###
        with open(os.path.join(src, batch, "deid_measurements.csv")) as f:
            header = f.readline().strip().split(",")

            # "Second Batch/deid_measurements.csv" actually contains all of the patient info (including the first batch)
            # Headers are different for "abnormals-deid and "Second Batch", so use header to identify
            if header == ['anon_patient_id', 'anon_accession_number', 'patient_sex', 'patient_age', 'patient_weight_kg', 'patient_height_cm', 'lv_ef_bullet', 'lv_ef_mod_a4c', 'lv_ef_mod_bp', 'lv_area_d_a4c', 'lv_area_s_a4c', 'lv_area_d_psax_pap', 'lv_area_s_psax_pap', 'lv_vol_d_bullet', 'lv_vol_s_bullet']:

                for line in f:
                    patient, accession, s, a, w, h, ef_bullet, *_ = line.strip().split(",")
                    try:
                        ef_bullet = float(ef_bullet)
                        ef[patient, accession] = ef_bullet
                        sex[patient, accession] = s
                        age[patient, accession] = a
                        try:
                            weight[patient, accession] = float(w)
                        except:
                            weight[patient, accession] = math.nan
                        try:
                            height[patient, accession] = float(h)
                        except:
                            height[patient, accession] = math.nan
                    except Exception as e:
                        print(type(e), e)
                        print("Invalid EF:", ef_bullet)

    ### Order points in trace ###
    for key in coordinates:
        for f in coordinates[key]:
            coordinates[key][f] = np.array(sorted(coordinates[key][f]))[:, 1:]

    ### Plot traces ###
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

    ### Assign views based on which traces are available ###
    for (p, a, i, f) in measurement:
        m = measurement[(p, a, i, f)]
        a4c = ("A4C" in m)
        psax = ("psax" in m or "PSAX" in m)

        if a4c and psax:
            raise ValueError("Both views")
        elif not a4c and not psax:
            # TODO: these are probably the ones in error (figure out what's wrong)
            print(p, a, i, f, " has no view", (p, a, i, f) in error)
        else:
            if a4c:
                view[(p, a, i)] = "A4C"
            elif psax:
                view[(p, a, i)] = "PSAX"

    for (p, a, i) in view:
        instance_of_view[(p, a)][view[(p, a, i)]].append(i)

    patients = sorted(set(p for (p, a) in ef))
    split = {p: int(hashlib.sha1(p.encode("utf-8")).hexdigest(), 16) % 10 for p in patients}

    ### Basic dataset statistics and plots ###
    print("Patients:", len(patients))
    print("Visits:", len(instance_of_view))
    print("Visits with A4C:", sum(x["A4C"] != [] for x in instance_of_view.values()))
    print("Visits with PSAX:", sum(x["PSAX"] != [] for x in instance_of_view.values()))
    visits = collections.Counter()
    for (p, a) in ef:
        visits[p] += 1
    s = {p: sex[p, a] for (p, a) in sex}
    print("Sex", collections.Counter(s.values()))
    for k in age:
        if age[k][-1] in "D":
            age[k] = "{:03d}Y".format(int(age[k][:-1]) // 365)
        if age[k][-1] in "W":
            age[k] = "{:03d}Y".format(int(age[k][:-1]) // 52)
        if age[k][-1] in "M":
            age[k] = "{:03d}Y".format(int(age[k][:-1]) // 12)
        assert len(age[k]) == 4 and age[k][-1] == "Y"
        age[k] = int(age[k][:-1])
    print("Age: ", np.mean(list(age.values())), "+/-", np.std(list(age.values())))
    print("EF: ", np.mean(list(ef.values())), "+/-", np.std(list(ef.values())))

    os.makedirs(os.path.join(dest, "fig"), exist_ok=True)

    fig = plt.figure(figsize=(1.5, 1.5))
    plt.hist(visits.values(), bins=range(max(visits.values())))
    plt.title("# Visits")
    plt.xlabel("# Visits")
    plt.ylabel("# Patients")
    plt.xlim([1, max(visits.values())])
    plt.tight_layout()
    plt.savefig(os.path.join(dest, "fig", "visits.pdf"))
    plt.close(fig)

    fig = plt.figure(figsize=(1.5, 1.5))
    plt.hist(age.values(), bins=range(max(age.values())))
    plt.title("Age")
    plt.xlabel("Age (years)")
    plt.ylabel("# Visits")
    plt.xlim([0, max(age.values())])
    plt.tight_layout()
    plt.savefig(os.path.join(dest, "fig", "age.pdf"))
    plt.close(fig)

    fig = plt.figure(figsize=(1.5, 1.5))
    plt.hist(ef.values(), bins=range(100))
    plt.title("Ejection Fraction")
    plt.xlabel("EF (%)")
    plt.ylabel("# Visits")
    plt.xlim([0, 100])
    plt.tight_layout()
    plt.savefig(os.path.join(dest, "fig", "ef.pdf"))
    plt.close(fig)

    fig = plt.figure(figsize=(1.5, 1.5))
    plt.scatter(*zip(*[(age[k], float(weight[k])) for k in age if k in weight]))
    plt.savefig(os.path.join(dest, "fig", "weight.pdf"))
    plt.close(fig)

    def save_videos(filename):
        # print(filename, flush=True)
        if filename in [
            os.path.join(src, "Second Batch", "dicom", "CR3dcb539-CR3dcb7b7.tgz"),  # Causes code to give "Aborted!"
            os.path.join(src, "Second Batch", "dicom", "CR3dcb53a-CR3dcb745.tgz"),  # Abort
            os.path.join(src, "Second Batch", "dicom", "CR3dcb53b-CR3dcb748.tgz"),  # Abort
        ]:
            return {}
        m = re.search(os.path.join(src, "Second Batch", "dicom", "(CR[0-9a-z]{7})-(CR[0-9a-z]{7}).tgz"), filename)
        # m = re.search(os.path.join(src, "({})".format("|".join(BATCHES)), "dicom", "(CR[0-9a-z]{7})-(CR[0-9a-z]{7}).tgz"), filename)
        assert m is not None
        
        # coord = collections.defaultdict(list)
        meta = collections.defaultdict(dict)
        patient, accession = m.groups()
        
        if (patient, accession) in metadata:
            return {}
        else:
            with tarfile.open(filename) as tf:
                for dicom in tf.getmembers():
                    if dicom.isfile():
                        InstanceNumber, SOPInstanceUID = os.path.basename(dicom.name).split("-")
                        assert len(InstanceNumber) == 6
            
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
                                    # A4Cs are upside-down; flip them
                                    video = video[:, :, ::-1, :]
                                    for f in c:
                                        c[f][:, 1] = video.shape[2] - c[f][:, 1] - 1
                                else:
                                    assert view[patient, accession, int(InstanceNumber)] == "PSAX"
                                video[1, :, :, :] = video[0, :, :, :]
                                video[2, :, :, :] = video[0, :, :, :]
    
                                # if not os.path.isfile(output["full"]):
                                #     echonet.utils.savevideo(output["full"], video, fps=fps)
                                # 
                                # for f in c:
                                #     if not os.path.isfile(output[f, "full"]):
                                #         frame = video[:, f, :, :].copy()
                                #         a, b = skimage.draw.polygon(c[f][:, 1], c[f][:, 0], (frame.shape[1], frame.shape[2]))
                                #         frame[2, a, b] = 255
                                #         PIL.Image.fromarray(frame.transpose((1, 2, 0))).save(output[f, "full"])
    
                                regions = ds.SequenceOfUltrasoundRegions
                                if len(regions) != 1:
                                    print("Found {} regions; expected 1.".format(len(regions)))
                                x0 = regions[0].RegionLocationMinX0
                                y0 = regions[0].RegionLocationMinY0
                                x1 = regions[0].RegionLocationMaxX1
                                y1 = regions[0].RegionLocationMaxY1

                                video = video[:, :, y0:(y1 + 1), x0:(x1 + 1)]
                                # if not os.path.isfile(output["crop"]):
                                #     echonet.utils.savevideo(output["crop"], video, fps=fps)
    
                                _, _, h, w = video.shape
                                video = video[:, :, :, ((w - h) // 2):(h + (w - h) // 2)]
                                if video.shape[2] != video.shape[3]:
                                    raise ValueError("Failed to make video square ({}, {})".format(video.shape[2], video.shape[3]))
    
                                for f in c:
                                    c[f] -= np.array([x0 + ((w - h) // 2), y0])
                                    c[f] = c[f] * 112 / np.array([video.shape[3], video.shape[2]])
                                    c[f] = c[f].astype(np.int64)
    
                                video = np.array(list(map(lambda x: cv2.resize(x, (112, 112), interpolation=cv2.INTER_AREA), video.transpose((1, 2, 3, 0))))).transpose((3, 0, 1, 2))
                                if not os.path.isfile(output["scale"]):
                                    echonet.utils.savevideo(output["scale"], video, fps=fps)
    
                                for f in c:
                                    if not os.path.isfile(output[f, "scale"]):
                                        frame = video[:, f, :, :]
                                        a, b = skimage.draw.polygon(c[f][:, 1], c[f][:, 0], (frame.shape[1], frame.shape[2]))
                                        frame[2, a, b] = 255
                                        PIL.Image.fromarray(frame.transpose((1, 2, 0))).save(output[f, "scale"])
    
                                # TODO: check dups
                                meta[patient, accession][InstanceNumber] = {"fps": fps, "region": (x0, y0, x1, y1)}
                                # coord[patient, accession, InstanceNumber].append((f, c[f]))
                            except Exception as e:
                                # TODO: exceptions are
                                # 1) <class 'ValueError'> axes don't match array
                                # 2) <class 'IndexError'> index 53 is out of bounds for axis 1 with size 53
                                #
                                # 1 is probably unfixable
                                # 2 can have different index, but is (almost?) always one off of the end; can probably just use last frame

                                print(filename, dicom.name)
                                print(type(e), e, flush=True)
                                print("", flush=True)
            return meta

    files = sorted(os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.join(src, "Second Batch")) for f in fn if os.path.splitext(f)[-1] == ".tgz")

    metadata_filename = os.path.join(dest, "metadata.pkl")
    try:
        with open(metadata_filename, "rb") as f:
            metadata = pickle.load(f)
    except:
        metadata = {}

    with tqdm.tqdm(total=len(files), desc="Saving videos") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            # for m in executor.map(save_videos, files):
            for m in map(save_videos, files):
                # TODO: check keys of c not already in coord
                metadata.update(m)

                if len(metadata) % 100 == 0 or len(metadata) == len(files):
                    with open(metadata_filename, "wb") as f:
                        pickle.dump(metadata, f)

                pbar.update()
    
    # TODO: transform coords
    for (p, a, i) in coordinates:
        if (p, a) in metadata and "{:06d}".format(i) in metadata[p, a]:
            (x0, y0, x1, y1) = metadata[p, a]["{:06d}".format(i)]["region"]
            for f in coordinates[p, a, i]:
                c = coordinates[p, a, i][f]

                h = y1 - y0 + 1
                w = x1 - x0 + 1

                c -= np.array([x0 + ((w - h) // 2), y0])
                c = c * 112 / h
                c = c.astype(np.int64)

                coordinates[p, a, i][f] = c

    for view in ["A4C", "PSAX"]:
        os.makedirs(os.path.join(dest, view), exist_ok=True)
        try:
            os.symlink(os.path.join("..", "Videos"), os.path.join(dest, view, "Videos"))
        except:
            pass

        with open(os.path.join(dest, view, "FileList.csv"), "w") as f:
            f.write("FileName,EF,Sex,Age,Weight,Height,Split\n")
            for (p, a) in ef:  # TODO: sort?
                for i in instance_of_view[(p, a)][view]:  # TODO: sort?
                    if (p, a) in metadata and "{:06d}".format(i) in metadata[p, a]:
                        f.write("{}-{}-{:06d}.avi,{},{},{},{},{},{}\n".format(p, a, i, ef[p, a], sex[p, a], age[p, a], weight[p, a], height[p, a], split[p]))

        # TODO: I think this can be symlinked?
        with open(os.path.join(dest, view, "VolumeTracings.csv"), "w") as f:
            # TODO: pad for systolic/distolic (whatever missing)
            f.write("FileName,X,Y,Frame\n")
            for (p, a) in ef:  # TODO: sort?
                for i in instance_of_view[(p, a)][view]:  # TODO: sort?
                    if (p, a) in metadata and "{:06d}".format(i) in metadata[p, a]:
                        if (p, a, i) in coordinates:
                            systolic = []
                            diastolic = []
                            for frame in coordinates[p, a, i]:
                                if measurement[p, a, i, frame] in ["LV_area_s_A4C_calc", "LV_area_s_psax_pap_calc"]:
                                    systolic.append(frame)
                                elif measurement[p, a, i, frame] in ["LV_area_d_A4C_calc", "LV_area_d_PSAX_pap_calc"]:
                                    diastolic.append(frame)
                                else:
                                    assert measurement[p, a, i, frame] == "INVALID"

                            print(systolic, diastolic)
                            for (label, frames) in [
                                ("Systolic", systolic),
                                ("Diastolic", diastolic),
                            ]:
                                if frames != []:
                                    frame = frames[0]
                                    for (x, y) in coordinates[p, a, i][frame]:
                                        f.write("{}-{}-{:06d}.avi,{},{},{}\n".format(p, a, i, x, y, frame))
                                else:
                                        f.write("{}-{}-{:06d}.avi,,,No {}\n".format(p, a, i, label))



if __name__ == "__main__":
    main()
