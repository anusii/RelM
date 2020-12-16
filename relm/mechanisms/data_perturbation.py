from .base import ReleaseMechanism
import numpy as np
from relm import backend
import scipy.sparse as sps


class SmallDB(ReleaseMechanism):
    """
    A offline Release Mechanism for answering a large number of queries.

    Args:
        epsilon: the privacy parameter
        data: a 1D array of the database in histogram format
        alpha: the relative error of the mechanism in range [0, 1]
    """

    def __init__(self, epsilon, data, alpha):

        super(SmallDB, self).__init__(epsilon)
        self.alpha = alpha

        if not type(alpha) is float:
            raise TypeError(f"alpha: alpha must be a float, found{type(alpha)}")

        if (alpha < 0) or (alpha > 1):
            raise ValueError(f"alpha: alpha must in [0, 1], found{alpha}")

        if not (data >= 0).all():
            raise ValueError(
                f"data: data must only non-negative values. Found {np.unique(data[data < 0])}"
            )

        if data.dtype == np.int64:
            data = data.astype(np.uint64)

        if data.dtype != np.uint64:
            raise TypeError(
                f"data: data must have either the numpy.uint64 or numpy.int64 dtype. Found {data.dtype}"
            )

        self.data = data
        self.db = None

    @property
    def privacy_consumed(self):
        if self._is_valid:
            return 0
        else:
            return self.epsilon

    def release(self, queries):
        """
        Releases differential private responses to queries.

        Args:
            queries: a 2D numpy array of queries in indicator format with shape (number of queries, db size)

        Returns:
            A numpy array of perturbed values.
        """

        self._check_valid()

        l1_norm = int(queries.shape[0] / (self.alpha ** 2)) + 1
        answers = queries.dot(self.data) / self.data.sum()

        error_str = (
            f"queries: queries must only contain 1s and 0s. Found {np.unique(queries)}"
        )

        if type(queries) is sps.csr.csr_matrix:
            if ((queries.data != 0) & (queries.data != 1)).any():
                raise ValueError(error_str)
            sparse_queries = queries.indices.astype(np.uint64)
            breaks = queries.indptr[1:].astype(np.uint64)

        else:
            if ((queries != 0) & (queries != 1)).any():
                raise ValueError(error_str)
            # store the indices of 1s of the queries in a flattened vector
            sparse_queries = np.concatenate(
                [np.where(queries[i, :])[0] for i in range(queries.shape[0])]
            ).astype(np.uint64)

            # store the indices of where each line ends in sparse_queries
            breaks = np.cumsum(queries.sum(axis=1).astype(np.uint64))

        db = backend.small_db(
            self.epsilon, l1_norm, len(self.data), sparse_queries, answers, breaks
        )

        self._is_valid = False

        return db
