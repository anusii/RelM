import numpy as np
import pandas as pd
import scipy.sparse as sps


class Histogram:
    """
    A class for converting a pandas Dataframe to histogram format as required by
    some DP mechanisms.

    Args:
        df: the Dataframe of the database
    """

    def __init__(self, df):
        _df = df.fillna(-999999)
        for col in df.columns:
            mask = _df[col] == -999999
            arr = _df[col].copy()
            arr[mask] = arr[mask].astype(df[col].dtype)
            _df[col] = arr

        _df["dummy"] = np.ones(len(_df))
        counts = _df.groupby(by=list(_df.columns[:-1])).count()
        columns = _df.columns[:-1]
        self.column_sets = []
        self.column_dict = dict((col, i) for i, col in enumerate(columns))
        self.column_incr = []

        incr = 1
        for column in columns:
            self.column_incr.append(incr)
            col_vals = set(pd.unique(_df[column]))
            self.column_sets.append(dict((y, x) for x, y in enumerate(col_vals)))
            incr *= len(col_vals)

        idxs = []
        vals = []

        for row in counts.index:
            if not isinstance(row, tuple):
                row = (row,)
            query = dict(zip(columns, row))
            idxs.append(self._get_idx(query))
            vals.append(counts.loc[row].dummy)

        self.idxs = np.array(idxs)
        self.vals = np.array(vals)
        self.size = incr

    def _get_idx(self, query):

        # Returns the index of the histogram database corresponding to the query.
        # The query must specify a value for each column of the underlying dataframe.

        # Args:
        #    query: a dictionary of column: value pairs specifying the query.

        # Returns:
        #    the index of the database histogram corresponding to the query

        idx = 0
        for col, val in query.items():
            i = self.column_dict[col]
            idx += self.column_sets[i][val] * self.column_incr[i]

        return idx

    def get_query_vector(self, query):
        """
        Returns the indices of the histogram database corresponding to the query.

        Args:
            query: a dictionary of (column: value) pairs specifying the query.

        Returns:
            the indices of the database histogram corresponding to the query
        """

        if not isinstance(query, list):
            query = [query]

        idxs_list = []
        rows_list = []
        vals_list = []
        for sub_query in query:
            sub_idxs = np.array([self._get_idx(sub_query)])
            for i, col in enumerate(self.column_dict.keys()):
                if sub_query.get(col, None) is None:
                    new_sub_idxs = (
                        np.arange(len(self.column_sets[i])) * self.column_incr[i]
                    )
                    sub_idxs = sub_idxs[:, None] + new_sub_idxs[None, :]
                    sub_idxs = sub_idxs.flatten()

            sub_rows = np.zeros_like(sub_idxs)
            sub_vals = np.ones_like(sub_idxs)

            idxs_list.append(sub_idxs)
            rows_list.append(sub_rows)
            vals_list.append(sub_vals)

        idxs = np.concatenate(idxs_list)
        rows = np.concatenate(rows_list)
        vals = np.concatenate(vals_list)

        vec = sps.csr_matrix((vals, (rows, idxs)), shape=(1, self.size))
        return vec

    def get_db(self):
        """
        Returns the database in histogram format.
        """

        db = np.zeros(self.size, dtype=np.uint64)
        db[self.idxs] = self.vals
        return db


class ObliviousHistogram(Histogram):
    def __init__(self, column_dict, column_sets, column_incr, db_size):
        self.column_dict = column_dict
        self.column_sets = column_sets
        self.column_incr = column_incr
        self.db_size = db_size

    def get_db(self):
        raise NotImplementedError
