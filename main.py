"""Run one experiment described by a configuration file.

    python main.py --f etc/benchmark.yaml

Most experiments are launched through run.py, which writes the configuration of a run from a base
file and command-line overrides and then calls this script.
"""
from Support.experiment import run


if __name__ == '__main__':
    run()
