import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from differential_privacy.mechanisms import GeometricMechanism

# Read the raw data.
data = pd.read_csv("pcr_testing_age_group_2020-03-09.csv")

# Compute the exact query responses.
exact_age_counts = data["age_group"].value_counts().sort_index()

# Create a differentially private release mechanism.
mechanism = GeometricMechanism(epsilon=0.1)
# Use the differentially private release mechanism to compute perturbed query responses:
perturbed_age_counts = mechanism.release(values=exact_age_counts.values)

# Display the exact query responses alongside the output of the
# differentially private release mechanism.
age_groups = np.sort(data["age_group"].unique())
age_ranges = np.array([a.lstrip("AgeGroup_") for a in age_groups])
df = pd.DataFrame(
    (
        {
            "Age Group": age_ranges,
            "Exact Counts": exact_age_counts,
            "Perturbed Counts": perturbed_age_counts,
        }
    )
)
print(df)

df.plot(x="Age Group", title="Test Counts by Age Group", kind="bar", rot=0)
plt.show()
