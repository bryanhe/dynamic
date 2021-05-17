#!/usr/bin/env python3

import os
import click
import pandas

@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False))
def main(src):
    for view in ["A4C", "PSAX"]:
        data = pandas.read_csv(os.path.join(src, view, "FileList.csv"), index_col=0)
        for split in range(10):
            os.makedirs(os.path.join(src, "{}_{}".format(view, split)), exist_ok=True)
            try:
                os.symlink(os.path.join("..", view, "Videos"), os.path.join(src, "{}_{}".format(view, split), "Videos"))
            except:
                pass
            try:
                os.symlink(os.path.join("..", view, "VolumeTracings.csv"), os.path.join(src, "{}_{}".format(view, split), "VolumeTracings.csv"))
            except:
                pass

            x = data.copy()
            def get_split(i):
                if i == split:
                    return "TEST"
                elif i == (split + 1) % 10:
                    return "VAL"
                return "TRAIN"

            x["Split"] = x["Split"].apply(get_split)
            x.to_csv(os.path.join(src, "{}_{}".format(view, split), "FileList.csv"))

if __name__ == "__main__":
    main()
