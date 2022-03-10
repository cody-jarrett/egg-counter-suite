"""
config.py

Author: John Raines (some tweaks made by Cody Jarrett)
https://github.com/jdraines/write-a-better-config.py
https://johndanielraines.medium.com/write-a-better-config-py-1a443cf5bb36

A custom config yaml may be provided in the following locations:

<VIRTUAL-ENVIRONMENT-ROOT>/.hello-world/config.yaml   # environment config
~/.hello-world/config.yaml                            # user config
/.hello-world/config.yaml                             # global config

First, the environment config location is checked for a config file. If not
found, then the user location is checked, and if no config file is found,
the global config location is checked. If no config file is found, the default
config.yaml included in this package will be used.

"""

import os
import sys
import yaml

from os.path import dirname, join

_env_configpath = join(sys.exec_prefix, ".egg-data-ana", "config.yaml")

_home_configpath = join(
    os.path.expanduser("~"), ".egg-data-ana", "config.yaml"
)

_global_configpath = "/.egg-data-ana/config.yaml"

_default_configpath = join(
    dirname(dirname(os.path.realpath(__file__))), "config.yaml"
)


def get_configpath() -> str:
    for path in [_env_configpath, _home_configpath, _global_configpath]:
        if os.path.exists(path):
            return path
    return _default_configpath


def get_config() -> dict:
    with open(get_configpath(), "rt") as f:
        return yaml.safe_load(f)


class _Config:

    # This method is defined to remind you that this is not a static class
    def __init__(self):
        pass

    @property
    def win_main_dir(self):
        return get_config()["win_main_dir"]

    @property
    def mac_main_dir(self):
        return get_config()["mac_main_dir"]

    @property
    def sub_dirs(self):
        return get_config()["sub_dirs"]

    @property
    def bad_rows(self):
        return get_config()["bad_rows"]

    @property
    def colors(self):
        return get_config()["colors"]

    @property
    def epoch_sep_settings(self):
        return get_config()["epoch_sep_settings"]

    @property
    def set_temps(self):
        return get_config()["set_temps"]

    @property
    def save_pic_flags(self):
        return get_config()["save_pic_flags"]

    @property
    def ec_log_file_name(self):
        return get_config()["ec_log_file_name"]

    @property
    def ec_data_load_settings(self):
        return get_config()["ec_data_load_settings"]

    @property
    def wrong_temp_sets(self):
        return get_config()["wrong_temp_sets"]


CONFIG = _Config()
