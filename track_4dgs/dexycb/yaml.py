"""DexYCB YAML reader. Calibration files contain Python tuples."""

import yaml


class DexYAMLLoader(yaml.SafeLoader):
    pass


def construct_tuple(loader, node):
    return loader.construct_sequence(node)


DexYAMLLoader.add_constructor("tag:yaml.org,2002:python/tuple", construct_tuple)


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=DexYAMLLoader)
