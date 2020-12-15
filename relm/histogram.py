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
        counts = df.groupby(by=list(df.columns[:-1])).count()

        df.fillna(-999999, inplace=True)
        df['dummy'] = np.ones(len(df))

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

        return idxs

#
# for row in one_day.itertuples(index=False):
#     query = dict(zip(columns[:-1], row[:-1]))
#
#     num_remove = np.random.randint(1, 3)
#     for col in np.random.choice(columns[:-1], size=num_remove, replace=False):
#         del query[col]
#
#     # ground truth
#     mask = np.ones(len(one_day)).astype(bool)
#     for col, val in query.items():
#         mask &= one_day[col] == val
#
#     # dp style db
#     _idxs = get_idxs(query, column_sets, column_incr, column_dict)
#     val = vals[np.isin(idxs, _idxs)].sum()
#
#     assert mask.sum() == val