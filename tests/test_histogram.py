import pandas as pd
import numpy as np
from relm.histogram import Histogram


def test_Histogram():
    df = pd.read_csv("examples/pcr_testing_age_group_2020-03-09.csv")
    columns = df.columns

    db = Histogram(df)

    for row in df.itertuples(index=False):
        query = dict(zip(columns[:-1], row[:-1]))

        num_remove = np.random.randint(1, len(columns))
        for col in np.random.choice(columns[:-1], size=num_remove, replace=False):
            del query[col]

        # ground truth
        mask = np.ones(len(df)).astype(bool)
        for col, val in query.items():
            mask &= df[col] == val

        # dp style db
        _idxs = db.get_idxs(query)
        val = db.vals[np.isin(db.idxs, _idxs)].sum()

        assert mask.sum() == val
