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

@app.route("/")
def _index():
    patients = sorted(os.listdir(os.path.join(DATA_DIR, "Videos")))
    return flask.render_template("index.html", todo=patients)

@app.route("/patient/<string:patient>", methods=["GET", "POST"])
def patient(patient):
    output = os.path.join(SAVE_DIR, "{}.pkl".format(patient))
    if flask.request.method == "GET":
        view = {}
        if os.path.isfile(output):
            try:
                with open(output, "r") as f:
                    for line in f:
                        video, v = line.split("\t")
                        view[video] = v
            except Exception as e:
                print(e)
                data = {}

        # videos = os.listdir(DATA_DIR)
        # videos = sorted(map(lambda v: os.path.splitext(v)[0], videos))
        # index = videos.index(video)
        # total = len(videos)
        # prev = None
        # if index != 0:
        #     prev = videos[index - 1]
        # next = None
        # if index + 1 < len(videos):
        #    next = videos[index + 1]
        # capture = cv2.VideoCapture(os.path.join(DATA_DIR, video + ".webm"))
        # fps = capture.get(cv2.CAP_PROP_FPS)
        # height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        videos = []
        for file in sorted(os.listdir(os.path.join(DATA_DIR, "Videos", patient))):
            if file not in ["full", "color", "complete.txt"]:
                videos.append(file)
        for i in range(len(videos)):
            if videos[i] in view:
                videos[i] = (videos[i], view[videos[i]])
            else:
                videos[i] = (videos[i], "-- Select View --")

        return flask.render_template("patient.html", patient=patient, videos=videos)
    else:
        data = flask.request.data
        data = data.strip().decode("utf-8").split("\n")
        print(data)
        data = [d.split(":") for d in data]
        print(data)
        data = {key: value for (key, value) in data}
        print(data)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        # TODO: save to temp loc and move
        with open(output, "wb") as f:
            pickle.dump(data, f)

        return ""


@app.route("/video/<string:patient>/<string:video>")
def video(patient, video):
    return flask.send_file(os.path.abspath(os.path.join(DATA_DIR, "Videos", patient, "full", video)))

if __name__ == "__main__":
    main()
