import numpy as np
import pandas as pd
import pathlib

from relm.mechanisms import GeometricMechanism

EPSILON = 2 ** -3
SENSITIVITY = 1.0

# ========================================================================================

# Read the raw data.
filename = "pcr_testing_age_group_2020-03-09.csv"
path = pathlib.Path(__file__, "..", filename).resolve()
data = pd.read_csv(path)

# Compute the exact query responses.
exact_counts = data["age_group"].value_counts().sort_index()

# ========================================================================================

# Usage
# Create a differentially private release mechanism.
mechanism = GeometricMechanism(epsilon=EPSILON, sensitivity=SENSITIVITY)

# Compute perturbed query responses.
perturbed_counts = mechanism.release(values=exact_counts.values)

# ========================================================================================

# Display the exact query responses alongside the perturbed query responses.
mechanism = GeometricMechanism(epsilon=EPSILON, sensitivity=SENSITIVITY)
perturbed_counts = mechanism.release(values=exact_counts.values)

# ----------------------------------------------------------------------------------------

age_groups = np.sort(data["age_group"].unique())
age_ranges = np.array([a.lstrip("AgeGroup_") for a in age_groups])
df = pd.DataFrame(
    {
        "Age Group": age_ranges,
        "Exact Counts": exact_counts,
        "Perturbed Counts": perturbed_counts,
    }
)
print(df)

# import matplotlib.pyplot as plt
# df.plot(x="Age Group", title="Test Counts by Age Group", kind="bar", rot=0)
# plt.show()
