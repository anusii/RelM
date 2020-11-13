import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from differential_privacy.mechanisms import GeometricMechanism

data = pd.read_csv("pcr_testing_age_group_2020-03-09.csv")

age_groups = np.sort(data['age_group'].unique())
raw_age_counts = data['age_group'].value_counts().sort_index()

epsilon = 0.1
mechanism = GeometricMechanism(epsilon)
dp_age_counts = mechanism.release(raw_age_counts.values)

age_ranges = np.array([a.lstrip('AgeGroup_') for a in age_groups])
df = pd.DataFrame(({'Age Group'      : age_ranges,
                    'Exact Count'    : raw_age_counts,
                    'Perturbed Count': dp_age_counts}))

df.plot(x='Age Group', title="Counts of Tests by Age Group", kind='bar', rot=0)
plt.show()
