import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from relm.mechanisms import AboveThreshold, SparseIndicator, SparseNumeric

EPSILON = 2 ** -3
SENSITIVITY = 1.0
THRESHOLD = 100.0

# ========================================================================================

# Read the raw data.
data = pd.read_csv("confirmed_cases_table4_likely_source.csv")

# Compute the exact first-hit index.
exact_counts = data["notification_date"].value_counts().sort_index()
values = exact_counts.values.astype(np.float64)
exact_first_hit = np.where(values >= THRESHOLD)[0][0]

# ========================================================================================

# Online (streaming) usage
# Create a differentially private release mechanism.
mechanism = AboveThreshold(
    epsilon=EPSILON, sensitivity=SENSITIVITY, threshold=THRESHOLD, monotonic=True
)

# Compute the differentially private first-hit index.
# Evaluate the values one at a time
for index, value in enumerate(values):
    hit = mechanism.release(values=values[index : index + 1])
    if hit is not None:
        dp_first_hit = index
        break

# ----------------------------------------------------------------------------------------

# Offline (batch) usage
# Create a differentially private release mechanism.
mechanism = AboveThreshold(
    epsilon=EPSILON, sensitivity=SENSITIVITY, threshold=THRESHOLD, monotonic=True
)

# Compute the differentially private first-hit index.
# Evaluate the values as a single batch
dp_first_hit = mechanism.release(values=values)

# ========================================================================================

TRIALS = 16

# Perform TRIALS independent experiments to capture the randomized nature of
# of the release mechanism
dp_first_hits = np.zeros(TRIALS, dtype=np.int)
for i in range(TRIALS):
    mechanism = AboveThreshold(
        epsilon=EPSILON, sensitivity=SENSITIVITY, threshold=THRESHOLD, monotonic=True
    )
    dp_first_hits[i] = mechanism.release(values=values)

# ----------------------------------------------------------------------------------------

# Display the exact first-hit index alongside the differentially private
# first-hit indices.
print("\tExact first-hit index: %i" % exact_first_hit)
print("\tDifferentially private first-hit indices:")
for i in range(TRIALS):
    print("\t\tExperiment %i: %i" % (i, dp_first_hits[i]))
