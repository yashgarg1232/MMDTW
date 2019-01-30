from scipy.spatial.distance import pdist
import numpy as np

class MMDTW:

    _ts1 = None  # Variates x Timestamps
    _ts2 = None  # Variates x Timestamps
    _metadata = None  # Variates x Variates

    _num_timestamps_1 = None
    _num_timestamps_2 = None
    _num_variates = None

    _distance = "cosine"

    def __init__(self, time_series_1=None, time_series_2=None, metadata=None):
        """
        Class constructor to initialize the time series and the associated metadata
        :param time_series_1: array, timestamps x variates
        :param time_series_2: array, timestamps x variates
        :param metadata: array, variates x variates
        """
        self._ts1 = time_series_1
        self._ts2 = time_series_2
        self._metadata = metadata

        self._num_timestamps_1 = self._ts1.shape[0]
        self._num_timestamps_2 = self._ts2.shape[0]

        self._num_variates = self._metadata.shape[0]

    def set_time_series1(self, time_series_1):
        self._ts1 = time_series_1

    def set_time_series2(self, time_series_2):
        self._ts1 = time_series_2

    def set_metadata(self, metadata):
        self._metadata = metadata

    def set_distance(self, distance):
        self._distance = distance

    def vdtw(self):
        cost_mat = np.full((self._num_timestamps_1, self._num_timestamps_2), np.inf)

        temp = np.zeros((2, self._num_variates))

        for t1 in range(0, self._num_timestamps_1):
            for t2 in range(0, self._num_timestamps_2):
                temp[0, :] = self._ts1[t1, :]
                temp[1, :] = self._ts2[t2, :]
                cost = pdist(temp, metric=self._distance)

                if t1 == 0 and t2 == 0:
                    cost_mat[t1, t2] = cost
                elif t1 == 0:
                    cost_mat[t1, t2] = cost + cost_mat[t1, t2 - 1]
                elif t2 == 0:
                    cost_mat[t1, t2] = cost + cost_mat[t1 - 1, t2]
                else:
                    cost_mat[t1, t2] = cost + np.min([cost_mat[t1 - 1, t2],
                                                      cost_mat[t1 - 1, t2 - 1],
                                                      cost_mat[t1, t2 - 1]])
        print(cost_mat[t1, t2])

    def idtw(self):
        pass

    def wdtw(self):
        pass

    def pdtw(self):
        pass


class MMException:
    def __init__(self, error_message):
        super().__init__(error_message)
