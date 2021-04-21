#!/usr/bin/env python3

import click
import os
import flask
import pickle
import cv2


app = flask.Flask(__name__)
DATA_DIR = None
SAVE_DIR = None
USERS = None

@click.command()
@click.argument("data_dir",
                type=click.Path(exists=True, file_okay=False))
@click.argument("save_dir",
                type=click.Path(file_okay=False))
@click.option('--host', default='0.0.0.0')
@click.option('-p', '--port', type=int, default=8000)
def main(data_dir, save_dir, host, port, users=["BH", "DD", "TT", "YD"]):
    global DATA_DIR
    global SAVE_DIR
    global USERS
    DATA_DIR = data_dir
    SAVE_DIR = save_dir
    USERS = users
    app.run(host=host, port=port, threaded=True, debug=True)

@app.route("/")
def _index():
    return index("")

@app.route("/<string:user>/")
def index(user):
    print(DATA_DIR)
    print(os.listdir(DATA_DIR))
    videos = os.listdir(DATA_DIR)
    videos = sorted(map(lambda v: os.path.splitext(v)[0], videos))
    try:
        pkl = set(map(lambda x: os.path.splitext(x)[0], os.listdir(os.path.join(SAVE_DIR, user))))
    except FileNotFoundError:
        pkl = set()
    todo = []
    done = []
    for v in videos:
        if v in pkl:
            done.append(v)
        else:
            todo.append(v)
    
    return flask.render_template("index.html", users=USERS, user=user, todo=todo, done=done)

@app.route("/<string:user>/<string:video>", methods=["GET", "POST"])
def label(user, video):
    output = os.path.join(SAVE_DIR, user, "{}.pkl".format(video))
    if flask.request.method == "GET":
        data = {}
        if os.path.isfile(output):
            try:
                with open(output, "rb") as f:
                    data = pickle.load(f)
                print(data)
            except:
                print("except")
                data = {}

        videos = os.listdir(DATA_DIR)
        videos = sorted(map(lambda v: os.path.splitext(v)[0], videos))
        index = videos.index(video)
        total = len(videos)
        prev = None
        if index != 0:
            prev = videos[index - 1]
        next = None
        if index + 1 < len(videos):
           next = videos[index + 1]
        capture = cv2.VideoCapture(os.path.join(DATA_DIR, video + ".webm"))
        fps = capture.get(cv2.CAP_PROP_FPS)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)

        labels = [("EF", ("Normal", "Slightly Reduced", "Moderately Reduced", "Severely Reduced")),
                  ("Interpretable", ("Yes", "Partial", "No"))]

        return flask.render_template("label.html", user=user, video=video, index=(index + 1), total=total, prev=prev, next=next, data=data, height=height, width=width, labels=labels)
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


@app.route("/video/<string:video>")
def video(video):
    return flask.send_file(os.path.abspath(os.path.join(DATA_DIR, video)))

if __name__ == "__main__":
    main()
