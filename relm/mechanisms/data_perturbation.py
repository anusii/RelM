from .base import ReleaseMechanism
import numpy as np
from relm import backend
import scipy.sparse as sps
from relm.mechanisms import SparseNumeric

# Imports for SmallDB debugging code
from itertools import combinations_with_replacement
from relm.mechanisms import ExponentialMechanism


class SmallDB(ReleaseMechanism):
    """
    A offline Release Mechanism for answering a large number of queries.

    Args:
        epsilon: the privacy parameter
        data: a 1D array of the database in histogram format
        alpha: the relative error of the mechanism in range [0, 1]
        db_size: the number of bins in the histogram representation of the database
        db_l1_norm: the number of records in the database

    """

    def __init__(self, epsilon, alpha, db_size, db_l1_norm):
        if not type(alpha) is float:
            raise TypeError(f"alpha: alpha must be a float, found{type(alpha)}")

        if (alpha < 0) or (alpha > 1):
            raise ValueError(f"alpha: alpha must in [0, 1], found{alpha}")

        super(SmallDB, self).__init__(epsilon)

        self.alpha = alpha
        self.db_size = db_size
        self.db_l1_norm = db_l1_norm

    @property
    def privacy_consumed(self):
        if self._is_valid:
            return 0
        else:
            return self.epsilon

    def release(self, values, queries):
        """
        Releases differential private responses to queries.

        Args:
            values: a numpy array of the exact query responses
            queries: a 2D numpy array of queries in indicator format with shape (number of queries, db size)

        Returns:
            A numpy array of perturbed values.
        """

        self._check_valid()

        l1_norm = int(queries.shape[0] / (self.alpha ** 2)) + 1

        sparse_queries, breaks = _process_queries(queries)

        db = backend.small_db(
            self.epsilon,
            l1_norm,
            self.db_size,
            self.db_l1_norm,
            sparse_queries,
            values,
            breaks,
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
        q_size: the number of queries answered by the mechanism
    """

    def __init__(self, epsilon, alpha, beta, q_size, db_size, db_l1_norm):
        if not type(alpha) in (float, np.float64):
            raise TypeError(f"alpha: alpha must be a float, found{type(alpha)}")

        if (alpha < 0) or (alpha > 1):
            raise ValueError(f"alpha: alpha must in [0, 1], found{alpha}")

        if type(q_size) is not int:
            raise TypeError(f"q_size: q_size must be an int. Found {type(q_size)}")

        if q_size <= 0:
            raise ValueError(f"q_size: q_size must be positive. Found {q_size}")

        super(PrivateMultiplicativeWeights, self).__init__(epsilon)

        self.alpha = alpha
        self.beta = beta
        self.q_size = q_size
        self.db_size = db_size
        self.db_l1_norm = db_l1_norm

        self.est_data = np.ones(self.db_size) / self.db_size
        self.learning_rate = self.alpha / 2

        cutoff = 4 * np.log(self.db_size) / (self.alpha ** 2)
        self.cutoff = int(cutoff)

        self.threshold = 18 * cutoff / (epsilon * self.db_l1_norm)
        self.threshold *= np.log(2 * self.q_size) + np.log(4 * cutoff / self.beta)

        self.sparse_numeric = SparseNumeric(
            epsilon,
            sensitivity=(1 / self.db_l1_norm),
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

        self.est_data *= np.exp(-r * self.learning_rate)
        self.est_data /= self.est_data.sum()

    def release(self, values, queries):
        """
        Returns private answers to the queries.

        Args:
            values: a numpy array of the exact query responses
            queries: a list of queries as 1D 1/0 indicator numpy arrays
        Returns:
            a numpy array of the private query responses
        """

        results = []
        for query, value in zip(queries, values):
            if type(query) is sps.csr.csr_matrix:
                query = np.asarray(query.todense()).flatten()

            est_answer = query.dot(self.est_data)

            error = value - est_answer
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
