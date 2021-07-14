#!/usr/bin/env python3

import pydicom
import io
import zipfile
import concurrent.futures
import PIL
import click
import cv2
import numpy as np
import os
import tqdm
import skimage.segmentation

import echonet


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False), default=os.path.join("/data", "bryanhe", "DICOM"))
@click.argument("dest", type=click.Path(), default="mrn")
def main(src, dest):
    os.makedirs(dest, exist_ok=True)
    if not os.path.isfile(os.path.join(dest, "video.csv")):
    # if True:
        with open(os.path.join(dest, "temp.csv"), "w") as f:
            f.write("VideoID,PatientID,Date,Time\n")
            for filename in sorted(os.listdir(src), key=lambda item: (int(item.partition(' ')[0]) if item[0].isdigit() else float('inf'), item)):
                print(filename)
                print("=" * len(filename))
                with zipfile.ZipFile(os.path.join(src, filename), 'r') as fusion_zip:
                    for exam_name in tqdm.tqdm(fusion_zip.namelist()):
                        if exam_name == "[Content_Types].xml":
                            continue
                        print("\t" + exam_name)
                        with fusion_zip.open(exam_name) as exam:
                            data = exam.read()
                        with zipfile.ZipFile(io.BytesIO(data), "r") as exam_zip:
                            assert exam_zip.namelist() == ['DICOM.zip', '[Content_Types].xml']
                            with exam_zip.open("DICOM.zip") as all_dicom:
                                data = all_dicom.read()
                            with zipfile.ZipFile(io.BytesIO(data), "r") as all_dicom_zip:
                                for dicom_name in sorted(all_dicom_zip.namelist(), key=lambda item: (int(item.partition(' ')[0]) if item[0].isdigit() else float('inf'), item)):
                                    if dicom_name == "[Content_Types].xml":
                                        continue
                                    print("\t\t" + dicom_name)
                                    with all_dicom_zip.open(dicom_name) as dicom:
                                        data = dicom.read()
                                    ds = pydicom.dcmread(io.BytesIO(data))

                                    assert dicom_name[:5] == "DICOM"
                                    assert dicom_name[-4:] == ".dcm"

                                    patient_id = ""
                                    try:
                                        patient_id = ds.OtherPatientIDs
                                    except AttributeError:
                                        pass

                                    if not isinstance(patient_id, pydicom.multival.MultiValue):
                                        patient_id = [patient_id]
                                    valid = [len(pid) == 8 and all(x in "0123456789" for x in pid) for pid in patient_id]
                                    if sum(valid) == 1:
                                        patient_id = patient_id[valid.index(True)]
                                    else:
                                        patient_id = ""

                                    date = ""
                                    try:
                                        # date = ds.PerformedProcedureStepStartDate
                                        date = ds.StudyDate
                                    except AttributeError:
                                        breakpoint()
                                        pass

                                    time = ""
                                    try:
                                        # time = ds.PerformedProcedureStepStartTime
                                        time = ds.StudyTime
                                    except AttributeError:
                                        pass

                                    # if dicom_name[5:-4] == "11":
                                    #     breakpoint()
                                    #     echonet.utils.savevideo("asd.avi", ds.pixel_array.transpose((3, 0, 1, 2)))
                                    f.write(",".join((dicom_name[5:-4], patient_id, date, time)) + "\n")
                                    f.flush()

        os.rename(os.path.join(dest, "temp.csv"), os.path.join(dest, "video.csv"))

    video = []
    with open(os.path.join(dest, "video.csv"), "r") as f:
        print(f.readline())
        for line in f:
            line = line.strip().split(",")
            assert line[0][0] != "0"
            line[0] = int(line[0])
            video.append(line)
            if len(line) != 4:
                print(line)
    video = sorted(video)

    with open(os.path.join(dest, "all_with_mrn.csv"), "w") as f:
        f.write("PatientID,Date\n")
        for (dicom, pid, date, time) in video:
            if pid != "":
                f.write("{},{}\n".format(pid, date))

    seen = set()
    with open(os.path.join(dest, "unique_mrn_and_date.csv"), "w") as f:
        f.write("PatientID,Date\n")
        for (dicom, pid, date, time) in video:
            if pid != "" and date != "":
                if (pid, date) not in seen:
                    f.write("{},{}\n".format(pid, date))
                seen.add((pid, date))

    with open(os.path.join(dest, "mrn_without_date.csv"), "w") as f:
        f.write("PatientID,Date\n")
        for (dicom, pid, date, time) in video:
            if pid != "" and date == "":
                f.write("{},{}\n".format(pid, date))

    len(set(map(lambda x: int(x[3:-5]), os.listdir("data/er_processed"))) - set(map(lambda x : x[0], video)))
    breakpoint()
    x = sorted(set(map(lambda x: int(os.path.splitext(x[3:])[0]), os.listdir("data/er_processed"))) - set(map(lambda x : x[0], video)))
    # print("\n".join(map(str, x)))
    import random
    random.shuffle(x)
    print("\n".join(map(str, x)))





if __name__ == "__main__":
    main()
