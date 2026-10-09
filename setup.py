from setuptools import find_namespace_packages, find_packages, setup

import os


with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()


pypi_build = os.environ.get("PYPI_BUILD", "").lower() in {"1", "true", "yes", "on"}

packages = ['track_4dgs'] + ["track_4dgs." + package for package in find_packages(where="track_4dgs")]
packages_matcha = []
packages_matcha += ["track_4dgs.matcha.dust3r"] + ["track_4dgs.matcha.dust3r." + package for package in find_namespace_packages(where="submodules/matcha/third_party/dust3r")]
packages_matcha += ["track_4dgs.matcha.geoaware_sc"] + ["track_4dgs.matcha.geoaware_sc." + package for package in find_namespace_packages(where="submodules/matcha/third_party/geoaware_sc")]
packages_matcha += ["track_4dgs.matcha.dift"] + ["track_4dgs.matcha.dift." + package for package in find_namespace_packages(where="submodules/matcha/third_party/dift")]
packages_matcha += ["track_4dgs.matcha.dinov2"] + ["track_4dgs.matcha.dinov2." + package for package in find_namespace_packages(where="submodules/matcha/third_party/dinov2")]
packages_uniflowmatch = ["uniflowmatch"] + ["uniflowmatch." + package for package in find_namespace_packages(where="submodules/UFM/uniflowmatch")]
packages_mvroma = ["mvroma"] + ["mvroma." + package for package in find_namespace_packages(where="submodules/MV-RoMa/src/mvroma")]
# Single module, not a package: packages= would install every *.py in that directory.
py_modules = ["track_4dgs.matcha.utils.category_list"]

setup(
    name="track_4dgs",
    version="0.8.1",
    author="yindaheng98",
    author_email="yindaheng98@gmail.com",
    url="https://github.com/yindaheng98/track-4dgs",
    description="Packaged Python point tracking utilities for 4D Gaussian Splatting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=packages + packages_matcha + packages_uniflowmatch + packages_mvroma,
    py_modules=py_modules,
    package_dir={
        "track_4dgs": "track_4dgs",
        "track_4dgs.matcha.dust3r": "submodules/matcha/third_party/dust3r",
        "track_4dgs.matcha.geoaware_sc": "submodules/matcha/third_party/geoaware_sc",
        "track_4dgs.matcha.dift": "submodules/matcha/third_party/dift",
        "track_4dgs.matcha.dinov2": "submodules/matcha/third_party/dinov2",
        "track_4dgs.matcha.utils": "submodules/matcha/matcha/utils",
        "uniflowmatch": "submodules/UFM/uniflowmatch",
        "mvroma": "submodules/MV-RoMa/src/mvroma",
    },
    install_requires=[
        "gaussian-splatting >= 2.3.8",
        "Pillow",
        "einops",
        "timm",
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
        # UniCeption (UFM encoder / info-sharing / prediction heads)
        "uniception @ git+https://github.com/castacks/UniCeption.git@ee7fa0b50d8295b3f0485e42c1d133fb534bfce7",
    ] if not pypi_build else []),
)
