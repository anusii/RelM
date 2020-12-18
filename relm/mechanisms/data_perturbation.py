from .base import ReleaseMechanism
import numpy as np
from relm import backend
import scipy.sparse as sps
from relm.mechanisms import SparseNumeric


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

        sparse_queries, breaks = _process_queries(queries)

        db = backend.small_db(
            self.epsilon, l1_norm, len(self.data), sparse_queries, answers, breaks
        )

        self._is_valid = False

        return db


class PrivateMultiplicativeWeights(ReleaseMechanism):
    """
    Secure implementation of the private Multiplicative Weights mechanism.
    This mechanism can be used to answer multiple linear queries.

    Args:
        epsilon: the privacy parameter to use
        data: a 1D numpy array of the underlying database
        alpha: the relative error of the mechanism
        num_queries: the number of queries answered by the mechanism
    """

    def __init__(self, epsilon, data, alpha, num_queries):
        super(PrivateMultiplicativeWeights, self).__init__(epsilon)

        if not type(alpha) in (float, np.float64):
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

        if type(num_queries) is not int:
            raise TypeError(
                f"num_queries: num_queries must be an int. Found {type(num_queries)}"
            )

        if num_queries <= 0:
            raise ValueError(
                f"num_queries: num_queries must be positive. Found {num_queries}"
            )

        self.l1_norm = data.sum()
        self.data = data / self.l1_norm
        self.data_est = np.ones(len(data)) / len(data)

        self.alpha = alpha
        self.learning_rate = self.alpha / 2

        # solve inequality of Theorem 4.14 (Dwork and Roth) for beta
        self.beta = epsilon * self.l1_norm * self.alpha ** 3
        self.beta /= 36 * np.log(len(data))
        self.beta -= np.log(num_queries)
        self.beta = np.exp(-self.beta) * 32 * np.log(len(data)) / (self.alpha ** 2)

        cutoff = 4 * np.log(len(data)) / (self.alpha ** 2)
        self.cutoff = int(cutoff)
        self.threshold = 18 * cutoff / (epsilon * self.l1_norm)
        self.threshold *= np.log(2 * num_queries) + np.log(4 * cutoff / self.beta)

        self.sparse_numeric = SparseNumeric(
            epsilon,
            sensitivity=(1 / self.l1_norm),
            threshold=self.threshold,
            cutoff=self.cutoff,
        )

        # this assumes that the l1 norm of the database is public

    @property
    def privacy_consumed(self):
        return self.sparse_numeric.privacy_consumed

    def update_weights(self, est_answer, noisy_answer, query):
        if noisy_answer < est_answer:
            r = query
        else:
            r = 1 - query

        self.data_est *= np.exp(-r * self.learning_rate)
        self.data_est /= self.data_est.sum()

    def release(self, queries):
        """
        Returns private answers to the queries.

        Args:
            queries: a list of queries as 1D 1/0 indicator numpy arrays
        Returns:
            a numpy array of the private query responses
        """

        results = []
        for query in queries:
            if type(query) is sps.csr.csr_matrix:
                query = np.asarray(query.todense()).flatten()

            true_answer = (query * self.data).sum()
            est_answer = (query * self.data_est).sum()

            error = true_answer - est_answer
            errors = np.array([error, -error])
            indices, release_values = self.sparse_numeric.release(errors)

            if len(indices) == 0:
                results.append(est_answer)
            else:
                noisy_answer = est_answer + (1 - 2 * indices[0]) * release_values[0]
                results.append(noisy_answer)
                self.update_weights(est_answer, noisy_answer, query)

        return np.array(results)


def _process_queries(queries):
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

    return sparse_queries, breaks
