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
