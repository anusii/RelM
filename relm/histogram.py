import numpy as np
import pandas as pd


class Histogram:
    """
    A class for converting a pandas Dataframe to histogram format as required by
    some DP mechanisms.

    Args:
        df: the Dataframe of the database
    """

    def __init__(self, df):
        df = df.copy()
        df.fillna(-999999, inplace=True)
        df["dummy"] = np.ones(len(df))
        counts = df.groupby(by=list(df.columns[:-1])).count()
        columns = df.columns[:-1]
        self.column_sets = []
        self.column_dict = dict((col, i) for i, col in enumerate(columns))
        self.column_incr = []

        incr = 1
        for column in columns:
            self.column_incr.append(incr)
            col_vals = set(pd.unique(df[column]))
            self.column_sets.append(dict((y, x) for x, y in enumerate(col_vals)))
            incr *= len(col_vals)

        idxs = []
        vals = []

        for row in counts.index:
            query = dict(zip(columns, row))
            idxs.append(self.get_idx(query))
            vals.append(counts.loc[row].dummy)

        self.idxs = np.array(idxs)
        self.vals = np.array(vals)
        self.size = incr

    def get_idx(self, query):
        """
        Returns the index of the histogram database corresponding to the query.
        The query must specify a value for each column of the underlying dataframe.

        Args:
            query: a dictionary of column: value pairs specifying the query.

        Returns:
            the index of the database histogram corresponding to the query
        """
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

        idxs = np.array([self.get_idx(query)])
        for i, col in enumerate(self.column_dict.keys()):
            if query.get(col, None) is None:
                new_idxs = np.arange(len(self.column_sets[i])) * self.column_incr[i]
                idxs = idxs[:, None] + new_idxs[None, :]
                idxs = idxs.flatten()

        vec = np.zeros(self.size)
        vec[idxs] = 1
        return vec

    def get_db(self):
        """
        Returns the database in histogram format.
        """

        db = np.zeros(self.size)
        db[self.idxs] = self.vals
        return db
