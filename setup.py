from setuptools import find_namespace_packages, find_packages, setup

import os


with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()


pypi_build = os.environ.get("PYPI_BUILD", "").lower() in {"1", "true", "yes", "on"}

packages = ['track_4dgs'] + ["track_4dgs." + package for package in find_packages(where="track_4dgs")]
packages_dust3r = ["track_4dgs.matcha.dust3r"] + ["track_4dgs.matcha.dust3r." + package for package in find_namespace_packages(where="submodules/matcha/third_party/dust3r")]
packages_geoaware_sc = ["track_4dgs.matcha.geoaware_sc"] + ["track_4dgs.matcha.geoaware_sc." + package for package in find_namespace_packages(where="submodules/matcha/third_party/geoaware_sc")]
packages_dift = ["track_4dgs.matcha.dift"] + ["track_4dgs.matcha.dift." + package for package in find_namespace_packages(where="submodules/matcha/third_party/dift")]
packages_dinov2 = ["track_4dgs.matcha.dinov2"] + ["track_4dgs.matcha.dinov2." + package for package in find_namespace_packages(where="submodules/matcha/third_party/dinov2")]
packages_utils = ["track_4dgs.matcha.utils"] + ["track_4dgs.matcha.utils." + package for package in find_namespace_packages(where="submodules/matcha/matcha/utils")]

setup(
    name="track_4dgs",
    version="0.7.1",
    author="yindaheng98",
    author_email="yindaheng98@gmail.com",
    url="https://github.com/yindaheng98/track-4dgs",
    description="Packaged Python point tracking utilities for 4D Gaussian Splatting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=packages + packages_dust3r + packages_geoaware_sc + packages_dift + packages_dinov2 + packages_utils,
    package_dir={
        "track_4dgs": "track_4dgs",
        "track_4dgs.matcha.dust3r": "submodules/matcha/third_party/dust3r",
        "track_4dgs.matcha.geoaware_sc": "submodules/matcha/third_party/geoaware_sc",
        "track_4dgs.matcha.dift": "submodules/matcha/third_party/dift",
        "track_4dgs.matcha.dinov2": "submodules/matcha/third_party/dinov2",
        "track_4dgs.matcha.utils": "submodules/matcha/matcha/utils",
    },
    install_requires=[
        "gaussian-splatting >= 2.3.8",
        "Pillow",
        "einops",
        "PyYAML",
        "numpy",
        "opencv-python",
        "warp-lang",
        # DIFT imports diffusers at module import. Stay below 0.36 so this does not
        # pull huggingface-hub 1.x, which breaks transformers<5.
        "diffusers>=0.30,<0.36",
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
