from setuptools import find_packages, setup

import os


with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()


pypi_build = os.environ.get("PYPI_BUILD", "").lower() in {"1", "true", "yes", "on"}


setup(
    name="track_4dgs",
    version="0.0.1",
    author="yindaheng98",
    author_email="yindaheng98@gmail.com",
    url="https://github.com/yindaheng98/track-4dgs",
    description="Packaged Python point tracking utilities for 4D Gaussian Splatting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    install_requires=[
        "gaussian-splatting >= 2.3.8",
        "Pillow",
    ] + ([
        # CoTracker3
        "cotracker @ git+https://github.com/facebookresearch/co-tracker.git@main",
        # VGGT and its dependencies
        "hydra-core",
        "omegaconf",
        "vggt @ git+https://github.com/facebookresearch/vggt.git@main",
        "lightglue @ git+https://github.com/jytime/LightGlue.git#egg=lightglue",
    ] if not pypi_build else []),
)
