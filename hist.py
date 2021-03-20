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

# bins = np.arange(102) - 0.5
# mul5 = ((bins % 5) > 4)
# color = ["gray" if m else "red" for m in mul5]
# fig = plt.figure(figsize=(3, 3))
# plt.hist(ef, bins=np.arange(101) - 0.5)
# plt.savefig("hist.pdf")
# plt.xlabel("EF")
# plt.ylabel("# Videos")
# plt.tight_layout()
# plt.close(fig)

ef = list(map(round, ef))
ef = collections.Counter(ef)
fig = plt.figure(figsize=(3, 3))
plt.bar(sorted(ef), [ef[x] for x in sorted(ef)], color=["red" if x % 5 == 0 else "gray" for x in sorted(ef)])
plt.xlabel("EF")
plt.ylabel("# Videos")
plt.tight_layout()
plt.savefig("hist.pdf")
plt.close(fig)
