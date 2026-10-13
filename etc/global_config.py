"""Global configuration, read once from the file given by --f when the package is imported."""
import argparse

import yaml

config = {}


def _init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default='etc/benchmark.yaml', help='configuration file')
    opt, _ = parser.parse_known_args()
    global config
    with open(opt.f, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)


def get_global_conf():
    _init()
    return config


config = get_global_conf()
