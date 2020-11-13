# differential-privacy
Implementations of differentially private release mechanisms

## Install

First of all, install rust: https://doc.rust-lang.org/book/ch01-01-installation.html.

Then:

```
$ git clone
$ cd differential-privacy
$ pip install .
```

Check the tests run:

```
$ pytest tests
```

That's it!

## Build docs

Install the docs dependencies:

```
$ pip install .[docs]
```

Build the docs:

```
sphinx-build -b html docs-source docs-build
```

The docs will now be in `docs-build/index.html`.

## Basic Usage
Read the raw data:
```python
import pandas as pd
data = pd.read_csv("pcr_testing_age_group_2020-03-09.csv")
```

Compute exact query responses:
```python
raw_age_counts = data["age_group"].value_counts().sort_index()
```

Create a differentially private release mechanism:
```python
from differential_privacy.mechanisms import GeometricMechanism
mechanism = GeometricMechanism(epsilon=0.1)
```

Apply the release mechanism to the exact query responses:
```python
dp_age_counts = mechanism.release(values=raw_age_counts.values)
```
