import math

import numpy as np
import scipy.stats

import relm.backend


def test_uniform_sampler():
    SCALES = [1.0, -1.0, 2.0, -.5, -math.tau, 1/math.e,
              1.2250738585072009e-308]
    for scale in SCALES:
        samples = relm.backend.sample_uniform(scale, 1_000_000)

        # Make sure samples have the correct sign/
        assert (np.copysign(1., samples) == math.copysign(1., scale)).all()

        # Take abs of scale and samples and verify it against
        # scipy.stats.uniform.
        scale = abs(scale)
        samples = np.abs(samples)
        score, pval = scipy.stats.kstest(samples, "uniform", args=(0, scale))
        assert pval > 0.001


def test_uniform_sampler_special_cases():
    for scale in [0., -0., float('inf'), -float('inf'), float('NaN')]:
        sample = relm.backend.sample_uniform(scale, 1)[0]
        assert sample == scale or (math.isnan(scale) and np.isnan(sample))
        # Preserves sign of negative zero:
        assert math.copysign(1., sample) == np.copysign(1., scale)
