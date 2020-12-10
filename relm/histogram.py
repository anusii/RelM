import numpy as np
import pandas as pd


class Histogram:

    def __init__(self, df):
        self.df = df.copy()
        counts = df.groupby(by=list(df.columns[:-1])).count()

        self.df.fillna(-999999, inplace=True)
        self.df['dummy'] = np.ones(len(self.df))

        columns = self.df.columns[:-1]

        self.column_sets = []
        self.column_dict = dict((col, i) for i, col in enumerate(columns))
        self.column_incr = []

        incr = 1
        for column in columns:
            self.column_incr.append(incr)
            col_vals = set(pd.unique(self.df[column]))
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
        idx = 0
        for col, val in query.items():
            i = self.column_dict[col]
            idx += self.column_sets[i][val] * self.column_incr[i]

        return idx

    def get_query_vector(self, query):

        idxs = np.array([self.get_idx(query)])
        for i, col in enumerate(self.column_dict.keys()):
            if query.get(col, None) is None:
                new_idxs = np.arange(len(self.column_sets[i])) * self.column_incr[i]
                idxs = idxs[:, None] + new_idxs[None, :]
                idxs = idxs.flatten()

        return idxs

