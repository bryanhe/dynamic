#!/usr/bin/env python3

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
