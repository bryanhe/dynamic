#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import collections

with open("predictions/predictions_original.csv") as f:
    ef_original = []
    f.readline()
    for l in f:
        ef_original.append(float(l.split(",")[-1]))

with open("predictions/predictions_mirror.csv") as f:
    ef_mirror = []
    f.readline()
    for l in f:
        ef_mirror.append(float(l.split(",")[-1]))

fig = plt.figure(figsize=(3, 3))
plt.hist([ef_mirror, ef_original], bins=range(0, 101, 5))
plt.legend(["Mirror", "Original"])
plt.xlabel("EF")
plt.ylabel("# Videos")
plt.tight_layout()
plt.savefig("hist.pdf")
plt.close(fig)

fig = plt.figure(figsize=(3, 3))
plt.scatter(ef_mirror, ef_original, s=1)
plt.xlabel("EF (mirror)")
plt.ylabel("EF (original)")
plt.tight_layout()
plt.savefig("scatter.pdf")
plt.close(fig)
