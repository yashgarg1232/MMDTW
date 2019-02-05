from scipy.spatial.distance import pdist
import numpy as np

class MMDTW:

    _ts1 = None  # Raw: Variates x Timestamps
    _ts2 = None  # Raw: Variates x Timestamps

    _ts1_p = None  # Projected: Variates x Timestamps
    _ts2_p = None  # Projected: Variates x Timestamps

    _metadata = None  # Variates x Variates

    _num_timestamps_1 = None
    _num_timestamps_2 = None
    _num_variates = None

    _distance = "cityblock"

    def __init__(self, time_series_1=None, time_series_2=None, metadata=None):
        """
        Class constructor to initialize the time series and the associated metadata
        :param time_series_1: array, timestamps x variates
        :param time_series_2: array, timestamps x variates
        :param metadata: array, variates x variates
        """
        if time_series_1 is not None:
            self._ts1 = time_series_1
            self._num_timestamps_1 = self._ts1.shape[0]

        if time_series_2 is not None:
            self._ts2 = time_series_2
            self._num_timestamps_2 = self._ts2.shape[0]

        if metadata is not None:
            self._metadata = metadata
            self._num_variates = self._metadata.shape[0]

    def set_time_series1(self, time_series_1):
        self._ts1 = time_series_1
        self._num_timestamps_1 = time_series_1.shape[0]

    def set_time_series2(self, time_series_2):
        self._ts1 = time_series_2
        self._num_timestamps_2 = time_series_2.shape[0]
    
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
                    # Starting case
                    cost_mat[t1, t2] = cost
                elif t1 == 0:
                    # Iterating over second time series
                    cost_mat[t1, t2] = cost + cost_mat[t1, t2 - 1]
                elif t2 == 0:
                    # Iterating over first time series
                    cost_mat[t1, t2] = cost + cost_mat[t1 - 1, t2]
                else:
                    # Computing the intermediate cost function
                    cost_mat[t1, t2] = cost + np.min([cost_mat[t1 - 1, t2],
                                                      cost_mat[t1 - 1, t2 - 1],
                                                      cost_mat[t1, t2 - 1]])
        print(cost_mat[t1, t2])

    def idtw(self, weights=None, projected=False):

        if weights is None or weights == []:
            weights = np.ones((self._num_variates,))

        final_cost = 0
        for v in range(0, self._num_variates):
            cost_mat = np.full((self._num_timestamps_1, self._num_timestamps_2), np.inf)

            temp = np.zeros((2, 1))

            for t1 in range(0, self._num_timestamps_1):
                for t2 in range(0, self._num_timestamps_2):
                    temp[0, 0] = self._ts1[t1, v] if not projected else self._ts1_p[t1, v]
                    temp[1, 0] = self._ts2[t2, v] if not projected else self._ts2_p[t1, v]
                    cost = pdist(temp, metric=self._distance)

                    if t1 == 0 and t2 == 0:
                        # Starting case
                        cost_mat[t1, t2] = cost
                    elif t1 == 0:
                        # Iterating over second time series
                        cost_mat[t1, t2] = cost + cost_mat[t1, t2 - 1]
                    elif t2 == 0:
                        # Iterating over first time series
                        cost_mat[t1, t2] = cost + cost_mat[t1 - 1, t2]
                    else:
                        # Computing the intermediate cost function
                        cost_mat[t1, t2] = cost + np.min([cost_mat[t1 - 1, t2],
                                                          cost_mat[t1 - 1, t2 - 1],
                                                          cost_mat[t1, t2 - 1]])
            final_cost += (cost_mat[t1, t2] * weights[v])
            print((cost_mat[t1, t2] * weights[v]))
        print(final_cost)

    def wdtw(self, invert_weigths=False):

        # Eignedecomposition to extract weights for Weighted Dynamic Time Warping
        eig_val, _ = np.linalg.eig(self._metadata)
        weights = np.abs(eig_val)
        # Inverting weights
        if invert_weigths:
            weights = np.reciprocal(weights[0:4])
        # Normalizing the weights to unity.
        weights /= np.sum(weights)

        return self.idtw(weights=weights)

    def pdtw(self, invert_weigths=True):
        eig_val, eig_vec = np.linalg.eig(self._metadata)
        weights = np.abs(eig_val)
        # Inverting weights
        if invert_weigths:
            weights = np.reciprocal(weights[0:4])
        # Normalizing the weights to unity.
        weights /= np.sum(weights)

        self._ts1_p = np.matmul(self._ts1, eig_vec)
        self._ts2_p = np.matmul(self._ts2, eig_vec)

        print(self._ts1_p.shape)

        # return self.idtw(weights=weights, projected=True)


class MMException:
    def __init__(self, error_message):
        super().__init__(error_message)
