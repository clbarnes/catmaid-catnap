#!/usr/bin/env python
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="catmaid-catnap",
    description="Experiments working with CATMAID and napari",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/clbarnes/catmaid-catnap",
    author="Chris L. Barnes",
    author_email="cb619@cam.ac.uk",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="neuroscience, connectomics, image, graph",
    packages=find_packages(include=["catnap*"]),
    python_requires=">=3.7, <4",
    install_requires=[
        "catpy>=2020.9.3",
        "coordinates",
        "h5py",
        "napari[all]",
        "numpy",
        "pandas",
        "scipy",
        "scikit-image",
        "StrEnum",
        "tables",
        "tqdm",
        "zarr",
    ],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    entry_points={
        "console_scripts": [
            "catnap=catnap.bin.view:main",
            "catnap-create=catnap.bin.create:main",
            "catnap-assess=catnap.bin.assess:main",
        ]
    },
    extras_require={},
)
