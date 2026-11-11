from setuptools import find_namespace_packages, find_packages, setup

import os


with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()


pypi_build = os.environ.get("PYPI_BUILD", "").lower() in {"1", "true", "yes", "on"}

packages = ['track_4dgs'] + ["track_4dgs." + package for package in find_packages(where="track_4dgs")]
packages_uniflowmatch = ["uniflowmatch"] + ["uniflowmatch." + package for package in find_namespace_packages(where="submodules/UFM/uniflowmatch")]
packages_mvroma = ["mvroma"] + ["mvroma." + package for package in find_namespace_packages(where="submodules/MV-RoMa/src/mvroma")]
packages_dot = ["dot"] + ["dot." + package for package in find_namespace_packages(where="submodules/dot/dot")]

setup(
    name="track_4dgs",
    version="1.10.2",
    author="yindaheng98",
    author_email="yindaheng98@gmail.com",
    url="https://github.com/yindaheng98/track-4dgs",
    description="Packaged Python point tracking utilities for 4D Gaussian Splatting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=packages + packages_uniflowmatch + packages_mvroma + packages_dot,
    package_dir={
        "track_4dgs": "track_4dgs",
        "uniflowmatch": "submodules/UFM/uniflowmatch",
        "mvroma": "submodules/MV-RoMa/src/mvroma",
        "dot": "submodules/dot/dot",
    },
    install_requires=[
        "gaussian-splatting >= 2.12.1",
        "Pillow",
        "einops",
        "einshape",
        "timm",
        "PyYAML",
        "numpy",
        "opencv-python",
        "warp-lang",
    ] + ([
        # CoTracker3
        "cotracker @ git+https://github.com/facebookresearch/co-tracker.git@main",
        # VGGT and its dependencies
        "hydra-core",
        "omegaconf",
        "vggt @ git+https://github.com/facebookresearch/vggt.git@main",
        "lightglue @ git+https://github.com/jytime/LightGlue.git#egg=lightglue",
        # UniCeption (UFM encoder / info-sharing / prediction heads)
        "uniception @ git+https://github.com/castacks/UniCeption.git@ee7fa0b50d8295b3f0485e42c1d133fb534bfce7",
    ] if not pypi_build else []),
)
