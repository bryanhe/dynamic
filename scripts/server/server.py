#!/usr/bin/env python3

import click
import os
import flask
import pickle
import cv2


app = flask.Flask(__name__)
DATA_DIR = None
SAVE_DIR = None

@click.command()
@click.argument("data_dir",
                type=click.Path(exists=True, file_okay=False))
@click.argument("save_dir",
                type=click.Path(file_okay=False))
@click.option('--host', default='0.0.0.0')
@click.option('-p', '--port', type=int, default=8000)
def main(data_dir, save_dir, host, port):
    global DATA_DIR
    global SAVE_DIR
    DATA_DIR = data_dir
    SAVE_DIR = save_dir
    app.run(host=host, port=port, threaded=True, debug=True)

VIEWS = ["A4C", "PSAX"]

@app.route("/")
def _index():
    patients = sorted(os.listdir(os.path.join(DATA_DIR, "Videos")))
    todo = []
    done = []
    for p in patients:
        if os.path.isfile(os.path.join(SAVE_DIR, "{}.tsv".format(p))):
            done.append(p)
        else:
            todo.append(p)
    print(patients[0])
    return flask.render_template("index.html", todo=todo, done=done, views=VIEWS, first=patients[0])

@app.route("/patient/<string:patient>", methods=["GET", "POST"])
def patient(patient):
    output = os.path.join(SAVE_DIR, "{}.tsv".format(patient))
    os.makedirs(SAVE_DIR, exist_ok=True)
    if flask.request.method == "GET":
        view = {}
        if os.path.isfile(output):
            try:
                with open(output, "r") as f:
                    for line in f:
                        video, v = line.strip().split("\t")
                        view[video] = v
            except Exception as e:
                print(e)
                data = {}

        patients = sorted(os.listdir(os.path.join(DATA_DIR, "Videos")))
        index = patients.index(patient)
        total = len(patients)
        prev = None
        if index != 0:
            prev = patients[index - 1]
        next = None
        if index + 1 < len(patients):
           next = patients[index + 1]

        videos = []
        for file in sorted(os.listdir(os.path.join(DATA_DIR, "Videos", patient))):
            if file not in ["full", "color", "complete.txt"]:
                videos.append(file)
                # capture = cv2.VideoCapture(os.path.join(DATA_DIR, "Videos", patient, "full", file))
                # fps = capture.get(cv2.CAP_PROP_FPS)
                # height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                # width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                # print(height, width, fps)
        for i in range(len(videos)):
            if videos[i] in view:
                videos[i] = (videos[i], view[videos[i]])
            else:
                videos[i] = (videos[i], "--Select View--")

        return flask.render_template("patient.html", patient=patient, videos=videos, index=index, total=total, prev=prev, next=next, views=(["--Select View--"] + VIEWS), view_of_video=view)
    else:
        data = flask.request.data
        data = data.decode("utf-8")
        # TODO: save to temp loc and move
        with open(output, "w") as f:
            f.write(data)

        return ""

@app.route("/view/<string:view>/<string:patient>", methods=["GET", "POST"])
def view_page(view, patient):

    if flask.request.method == "GET":
        patients = sorted(os.listdir(os.path.join(DATA_DIR, "Videos")))
        index = patients.index(patient)
        total = len(patients)
        prev = None
        if index >= 10:
            prev = patients[index - 10]
        next = None
        if index + 10 < len(patients):
           next = patients[index + 10]

        patient_videos = []
        for patient in patients[index:(index + 10)]:
            output = os.path.join(SAVE_DIR, "{}.tsv".format(patient))
            videos = []
            if os.path.isfile(output):
                try:
                    with open(output, "r") as f:
                        for line in f:
                            video, v = line.strip().split("\t")
                            if v == view:
                                videos.append(video)
                except Exception as e:
                    print(e)
                    data = {}
            patient_videos.append((patient, videos))

        # videos = []
        # for file in sorted(os.listdir(os.path.join(DATA_DIR, "Videos", patient))):
        #     if file not in ["full", "color", "complete.txt"]:
        #         videos.append(file)
        #         # capture = cv2.VideoCapture(os.path.join(DATA_DIR, "Videos", patient, "full", file))
        #         # fps = capture.get(cv2.CAP_PROP_FPS)
        #         # height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #         # width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        #         # print(height, width, fps)
        # for i in range(len(videos)):
        #     if videos[i] in view:
        #         videos[i] = (videos[i], view[videos[i]])
        #     else:
        #         videos[i] = (videos[i], "--Select View--")

        return flask.render_template("view.html", patient=patient, index=index, total=total, prev=prev, next=next, views=(["--Select View--"] + VIEWS), patient_videos=patient_videos, view=view)
    else:
        data = flask.request.data
        data = data.decode("utf-8")
        # TODO: save to temp loc and move
        with open(output, "w") as f:
            f.write(data)

        return ""


@app.route("/video/<string:patient>/<string:video>")
def video(patient, video):
    return flask.send_file(os.path.abspath(os.path.join(DATA_DIR, "Videos", patient, "full", video)))

if __name__ == "__main__":
    main()
