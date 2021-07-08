#!/usr/bin/env python3

import skimage.feature
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
import collections

import echonet


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False))
@click.argument("dest", type=click.Path(file_okay=False))
@click.option("--patients", multiple=True)
def main(src, dest, patients):
    try:
        root = {}
        with open(os.path.join(dest, "root.tsv"), "r") as f:
            for line in f:
                p, path = line.strip().split("\t")
                root[p] = path
    except FileNotFoundError:
        root = {}
        for batch in os.listdir(src):
            if batch in ["$RECYCLE.BIN", "System Volume Information"]:
                continue

            for patient in os.listdir(os.path.join(src, batch)):
                if not os.path.isdir(os.path.join(src, batch, patient)) or patient == "Viewer":
                    continue

                # print(batch, patient)
                p = patient.split("_")[0]
                assert patient == "{}_000_{}_".format(p, p) or patient == "{}_{}_".format(p, p)
                if p in root:
                    # Patient appears multiple times
                    # (probably across batches, but potentially just with missing _000)
                    print(*root[p])
                    print(batch, patient)

                    prev = collections.defaultdict(int)
                    for filename in os.listdir(os.path.join(src, *root[p])):
                        prev[filename] = os.path.getsize(os.path.join(src, *root[p], filename))

                    curr = collections.defaultdict(int)
                    for filename in os.listdir(os.path.join(src, batch, patient)):
                        curr[filename] = os.path.getsize(os.path.join(src, batch, patient, filename))

                    filenames = set().union(prev.keys(), curr.keys())

                    if all(prev[f] == curr[f] for f in filenames):
                        # print("Same")
                        pass
                    elif all(prev[f] >= curr[f] for f in filenames):
                        # print("Old is bigger")
                        pass
                    elif all(prev[f] <= curr[f] for f in filenames):
                        # print("New is bigger")
                        root[p] = (batch, patient)
                    else:
                        print("Mixed")
                        assert p == "90084209"
                        if len(os.listdir(os.path.join(src, batch, patient))) > len(os.listdir(os.path.join(src, *root[p]))):
                            root[p] = (batch, patient)

                    print()

                else:
                    root[p] = (batch, patient)

        for p in root:
            root[p] = os.path.join(*root[p])

        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, "root.tsv"), "w") as f:
            for p in sorted(root):
                f.write("{}\t{}\n".format(p, root[p]))

    logo = PIL.Image.open(os.path.join(os.path.dirname(__file__), "phillips.png"))
    logo = np.array(logo)

    if patients == ():
        patients = sorted(root.keys())
    print(patients)
    for p in tqdm.tqdm(patients):
        os.makedirs(os.path.join(dest, p), exist_ok=True)

        def save_file(filename):
            if os.path.isfile(os.path.join(dest, p, "full", filename[3:] + ".webm")) or \
               os.path.isfile(os.path.join(dest, p, "color", filename[3:] + ".webm")):
               # skip if done
               return

            ds = pydicom.dcmread(os.path.join(src, root[p], filename), force=True)

            try:
                video = ds.pixel_array
            except AttributeError:
                return
            if len(video.shape) in (2, 3):
                return

            res = skimage.feature.match_template(video[0, :, :, :], logo)
            i, j, _ = np.unravel_index(np.argmax(res), res.shape)

            if len(ds.SequenceOfUltrasoundRegions) != 1:
                print("Found {} regions; expected 1.".format(len(ds.SequenceOfUltrasoundRegions)))
                # breakpoint()
            # assert len(ds.SequenceOfUltrasoundRegions) == 1
            region = ds.SequenceOfUltrasoundRegions[0]
            x0 = region.RegionLocationMinX0
            y0 = region.RegionLocationMinY0
            x1 = region.RegionLocationMaxX1
            y1 = region.RegionLocationMaxY1

            video = pydicom.pixel_data_handlers.util.convert_color_space(video, ds.PhotometricInterpretation, "RGB")
            video = video.transpose((3, 0, 1, 2))

            small = video[:, :, y0:(y1 + 1), x0:(x1 + 1)]
            _, _, h, w = small.shape
            small = small[:, :, :, ((w - h) // 2):(h + (w - h) // 2)]
            small = np.array(list(map(lambda x: cv2.resize(x, (112, 112), interpolation=cv2.INTER_AREA), small.transpose((1, 2, 3, 0))))).transpose((3, 0, 1, 2))

            if i > 250:
                # Upside-down
                video = video[:, :, ::-1, :]
                small = small[:, :, ::-1, :]

            fps = 1000 / float(ds.FrameTime)

            assert filename[:3] == "IMG"
            if ds.UltrasoundColorDataPresent == 0:
                echonet.utils.savevideo(os.path.join(dest, p, filename[3:] + ".webm"), small, fps)
                os.makedirs(os.path.join(dest, p, "full"), exist_ok=True)
                echonet.utils.savevideo(os.path.join(dest, p, "full", filename[3:] + ".webm"), video, fps)
            else:
                os.makedirs(os.path.join(dest, p, "color"), exist_ok=True)
                echonet.utils.savevideo(os.path.join(dest, p, "color", filename[3:] + ".webm"), video, fps)

        filenames = sorted(os.listdir(os.path.join(src, root[p])))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # for filename in tqdm.tqdm(executor.map(save_file, filenames), total=len(filenames), leave=False):
            for filename in tqdm.tqdm(map(save_file, filenames), total=len(filenames), leave=False):
                pass

        with open(os.path.join(dest, p, "complete"), "w") as f:
            # write file to mark complete
            f.write(p)
            f.write("\n")

    return
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
                        measurement[key] = "INVALID"
                        error.add(key)
                else:
                    measurement[key] = m

        ### Read metadata ###
        with open(os.path.join(src, batch, "deid_measurements.csv")) as f:
            header = f.readline().strip().split(",")
            # assert header == ['anon_patient_id', 'anon_accession_number', 'lv_ef_bullet', 'lv_ef_mod_a4c', 'lv_ef_mod_bp', 'lv_area_d_a4c', 'lv_area_s_a4c', 'lv_area_d_psax_pap', 'lv_area_s_psax_pap', 'lv_vol_d_bullet', 'lv_vol_s_bullet']
            # "Second Batch/deid_measurements.csv" actually contains all of the patient info (including the first batch)
            # Headers are different, so just use the complete header
            if header == ['anon_patient_id', 'anon_accession_number', 'patient_sex', 'patient_age', 'patient_weight_kg', 'patient_height_cm', 'lv_ef_bullet', 'lv_ef_mod_a4c', 'lv_ef_mod_bp', 'lv_area_d_a4c', 'lv_area_s_a4c', 'lv_area_d_psax_pap', 'lv_area_s_psax_pap', 'lv_vol_d_bullet', 'lv_vol_s_bullet']:

                for line in f:
                    patient, accession, s, a, w, h, ef_bullet, *_ = line.strip().split(",")
                    try:
                        ef_bullet = float(ef_bullet)
                        ef[patient, accession] = ef_bullet
                        sex[patient, accession] = s
                        age[patient, accession] = a
                        weight[patient, accession] = w
                        height[patient, accession] = h
                    except:
                        print("Invalid EF:",ef_bullet)

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
            # TODO: these are probably the ones in errors (figure out what's wrong)
            print(p, a, i, f, " has no view")
        else:
            if a4c:
                view[(p, a, i)] = "A4C"
            elif psax:
                view[(p, a, i)] = "PSAX"

    for (p, a, i) in view:
        instance_of_view[(p, a)][view[(p, a, i)]].append(i)

    patients = sorted(set(p for (p, a) in ef))
    # index = {p: i for (i, p) in enumerate(patients)}
    split = {p: int(hashlib.sha1(p.encode("utf-8")).hexdigest(), 16) % 10 for p in patients}
    # for p in index:
    #     split[p] = (index[p] % 10)
    # with open(os.path.join(dest, "FileList.csv"), "w") as f:
    #     f.write("FileName,EF,Split\n")
    #     for (p, a) in ef:
    #         for i in instance_of_view[(p, a)]["A4C"]:
    #             f.write("{}-{}-{:06d}.avi,{},{}\n".format(p, a, i, ef[(p, a)], split[p]))
    
    def get_metadata(filename):
        m = re.search(os.path.join(src, "Second Batch", "dicom", "(CR[0-9a-z]{7})-(CR[0-9a-z]{7}).tgz"), filename)
        # m = re.search(os.path.join(src, "({})".format("|".join(BATCHES)), "dicom", "(CR[0-9a-z]{7})-(CR[0-9a-z]{7}).tgz"), filename)
        assert m is not None
        
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
            
                        with tf.extractfile(dicom.name) as f:
                            ds = pydicom.dcmread(f)
            
                        fps = None
                        try:
                            fps = ds.CineRate  # TODO CineRate frequently missing
                        except:
                            pass
            
                        regions = ds.SequenceOfUltrasoundRegions
                        if len(regions) != 1:
                            print("Found {} regions; expected 1.".format(len(regions)))
                        x0 = regions[0].RegionLocationMinX0
                        y0 = regions[0].RegionLocationMinY0
                        x1 = regions[0].RegionLocationMaxX1
                        y1 = regions[0].RegionLocationMaxY1

                        # TODO: check for duplicates
                        # if (patient, accession, InstanceNumber) in meta:
                        #     print(filename, "has multiple copies", flush=True)
                        # assert (patient, accession, InstanceNumber) not in meta
                        meta[patient, accession][InstanceNumber] = {"fps": fps, "region": (x0, y0, x1, y1)}
            return meta

    def save_videos(filename):
        # print(filename, flush=True)
        if filename in [
            "/scratch/users/bryanhe/pediatric_echos/Second Batch/dicom/CR3dcb539-CR3dcb7b7.tgz",
            "/scratch/users/bryanhe/pediatric_echos/Second Batch/dicom/CR3dcb53a-CR3dcb745.tgz",
            "/scratch/users/bryanhe/pediatric_echos/Second Batch/dicom/CR3dcb53b-CR3dcb748.tgz"
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

    # with tqdm.tqdm(total=len(files), desc="Reading metadata from DICOMs") as pbar:
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    #         for m in executor.map(get_metadata, files):
    #         # for m in map(get_metadata, files):
    #             # TODO: check keys of c not already in metadata
    #             metadata.update(m)

    #             if len(metadata) % 100 == 0:
    #                 with open(metadata_filename, "wb") as f:
    #                     pickle.dump(metadata, f)

    #             pbar.update()


    # with open(metadata_filename, "wb") as f:
    #     pickle.dump(meta, f)

    with tqdm.tqdm(total=len(files), desc="Saving videos") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            for m in executor.map(save_videos, files):
            # for m in map(save_videos, files):
                # TODO: check keys of c not already in coord
                metadata.update(m)

                if len(metadata) % 100 == 0:
                    with open(metadata_filename, "wb") as f:
                        pickle.dump(metadata, f)

                pbar.update()

    with open(metadata_filename, "wb") as f:
        pickle.dump(metadata, f)
    
    for view in ["A4C", "PSAX"]:
        os.makedirs(os.path.join(dest, view), exist_ok=True)
        try:
            os.symlink(os.path.join("..", "Videos"), os.path.join(dest, view, "Videos"))
        except:
            pass

        with open(os.path.join(dest, view, "FileList.csv"), "w") as f:
            f.write("FileName,EF,Sex,Age,Weight,Height,Split\n")
            for (p, a) in ef:
                for i in instance_of_view[(p, a)][view]:
                    if (p, a) in metadata and "{:06d}".format(i) in metadata[p, a]:
                        f.write("{}-{}-{:06d}.avi,{},{},{},{},{},{}\n".format(p, a, i, ef[p, a], sex[p, a], age[p, a], weight[p, a], height[p, a], split[p]))

        # with open(os.path.join(dest, view, "VolumeTracings.csv"), "w") as f:
        #     f.write("FileName,X,Y,Frame\n")
        #     for (p, a) in ef:
        #         for i in instance_of_view[(p, a)][view]:
        #             if (p, a, "{:06d}".format(i)) in coord:
        #                 for (frame, c) in coord[p, a, "{:06d}".format(i)]:
        #                     for (x, y) in c:
        #                         f.write("{}-{}-{:06d}.avi,{},{},{}\n".format(p, a, i, x, y, frame))



if __name__ == "__main__":
    main()
