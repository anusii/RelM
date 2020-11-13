# differential-privacy
Implementations of differentially private release mechanisms

## Install

First of all, install rust: https://doc.rust-lang.org/book/ch01-01-installation.html.

Then:

```
$ git clone https://github.com/anusii/differential-privacy
$ cd differential-privacy
$ pip install .
```

That's it!

## Run tests

Install the test dependencies:

```
pip install .[tests]
```

Check the tests run:

```
$ pytest tests
```


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
