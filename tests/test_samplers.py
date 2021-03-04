import numpy as np
import scipy.stats

import relm.backend


def test_uniform_sampler():
    samples = relm.backend.sample_uniform(1.0, 10_000_000)
    score, pval = scipy.stats.kstest(samples, 'uniform')
    assert pval > 0.001
