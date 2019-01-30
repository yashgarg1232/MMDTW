from codec import MMDTW

import numpy as np

ts1 = np.loadtxt('./data/1.csv', delimiter=',', dtype=float)
ts2 = np.loadtxt('./data/2.csv', delimiter=',', dtype=float)
metadata = np.loadtxt('./data/metadata.csv', delimiter=',', dtype=float)

print(ts1.shape)
print(ts2.shape)
print(metadata.shape)


mmdtw = MMDTW(time_series_1=ts1, time_series_2=ts2, metadata=metadata)

mmdtw.vdtw()