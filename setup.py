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
install_requires = ["numpy>=1.14.5"]

extras_requires = {
    "docs": ["Sphinx==3.3.0", "sphinx-rtd-theme==0.5.0"],
    "tests": ["pytest-benchmark==3.2.3", "pytest==6.0.1", "scipy==1.5.2"]
}

setup(
    name="differential-privacy",
    version="0.1.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Rust",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
    ],
    packages=["differential_privacy"],
    rust_extensions=[RustExtension("differential_privacy.backend")],
    install_requires=install_requires,
    extras_require=extras_requires,
    setup_requires=setup_requires,
    include_package_data=True,
    zip_safe=False,
)
