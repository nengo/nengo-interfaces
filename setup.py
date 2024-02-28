# Automatically generated by nengo-bones, do not edit this file directly

import io
import pathlib
import runpy

try:
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py"
    )


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = pathlib.Path(__file__).parent
version = runpy.run_path(str(root / "nengo_interfaces" / "version.py"))["version"]

install_req = [
    "nengo>=3.0.0",
]
docs_req = []
optional_req = []
tests_req = [
    "pytest>=4.3.0",
]

setup(
    name="nengo-interfaces",
    version=version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    url="https://www.nengo.ai/nengo-interfaces",
    include_package_data=False,
    license="Free for non-commercial use",
    description="Simplified external input and output communication",
    long_description=read("README.rst", "CHANGES.rst"),
    zip_safe=False,
    install_requires=install_req,
    extras_require={
        "all": docs_req + optional_req + tests_req,
        "docs": docs_req,
        "optional": optional_req,
        "tests": tests_req,
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Alpha",
        "Framework :: Nengo",
        "Intended Audience :: Science/Research",
        "License :: Free for non-commercial use",
        "Operating System :: OS Independent",
        "Programming Language :: Python ",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering ",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
