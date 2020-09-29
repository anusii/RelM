#!/usr/bin/env python
import sys

from setuptools import setup

try:
    from setuptools_rust import RustExtension
except ImportError:
    import subprocess

    errno = subprocess.call([sys.executable, "-m", "pip", "install", "setuptools-rust"])
    if errno:
        print("Please install setuptools-rust package")
        raise SystemExit(errno)
    else:
        from setuptools_rust import RustExtension

setup_requires = ["setuptools-rust>=0.10.1", "wheel"]
install_requires = ["numpy", "numba", "pytest", "pytest-benchmark"]

setup(
    name="differential-privacy",
    version="0.1.0",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Rust",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
    ],
    packages=["differential_privacy"],
    rust_extensions=[RustExtension("differential_privacy.differential_privacy")],
    install_requires=install_requires,
    setup_requires=setup_requires,
    include_package_data=True,
    zip_safe=False,
)