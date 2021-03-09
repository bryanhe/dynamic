#!/usr/bin/env python3

import random
import os
import pandas
import collections
def main(path="/oak/stanford/groups/jamesz/bryanhe/echocardiograms/"):
    x = pandas.read_csv(os.path.join(path, "A4c-StudyID-Filename.csv"))
    y = pandas.read_csv(os.path.join(path, "StudyID-EF.csv"))
    
    x = x.drop("Unnamed: 0", axis=1)
    x = x.set_index("StudyIdk")
    y = y.set_index("StudyIdk")
    
    z = x.join(y, how="inner")
    # z = z.set_index("V1")
    
    
    count = collections.Counter(z["V1"])
    for k in count:
        if count[k] != 1:
            print(k)
            print(z[z["V1"] == k])
    
    
    files = os.listdir(os.path.join(path, "Videos"))
    
    f = sorted(z["V1"].tolist())
    random.seed(0)
    random.shuffle(f)
    index = {f: i for (i, f) in enumerate(f)}
    def split(code):
        i = index[code]
        if i < 0.90 * len(index):
            return "TRAIN"
        elif i < 0.95 * len(index):
            return "VAL"
        else:
            return "TEST"
    
    
    z["Split"] = z["V1"].apply(split)
    z["V1"] = z["V1"].apply("ds_{}.avi".format)
    labeled = z["V1"].tolist()
    assert len(labeled) == len(set(labeled))
    z.columns = ["FileName", "EF", "Split"]
    z = z.set_index("FileName")
    with open(os.path.join(path, "FileList.csv"), "w") as f:
        f.write(z.loc[sorted(set(labeled).intersection(set(files)))].to_csv())


if __name__ == "__main__":
    main()
