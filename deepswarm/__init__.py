# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import argparse
import operator
import os
import sys

from pathlib import Path
from shutil import copyfile
from yaml import load, Loader


# Create argument parser which allows users to pass a custom settings file name
# If the user didn't pass a custom script name then use sys.argv[0]
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--settings_file_name', default=os.path.basename(sys.argv[0]),
    help='Settings file name. The default value is the name of invoked script without the .py extenstion')
args, _ = parser.parse_known_args()

# Retrieve filename without the extension
filename = os.path.splitext(args.settings_file_name)[0]

# If mnist.yaml doesn't exist it means that the package was installed via pip in
# which case we should use the current working directory as the base path
base_path = Path(os.path.dirname(os.path.dirname(__file__)))
if not (base_path / 'settings' / 'mnist.yaml').exists():
    module_path = base_path

    # Change the base path to the current working directory
    base_path = Path(os.getcwd())
    settings_directory = (base_path / 'settings')

    # Create settings directory if it doesn't exist
    if not settings_directory.exists():
        settings_directory.mkdir()

    # If default settings file doesn't exist, copy one from the module directory
    module_default_config = module_path / 'settings/default.yaml'
    settings_default_config = settings_directory / 'default.yaml'
    if not settings_default_config.exists() and module_default_config.exists():
        copyfile(module_default_config, settings_default_config)

# As the base path is now configured we try to load configuration file
# associated with the filename
settings_directory = base_path / 'settings'
settings_file_path = Path(settings_directory, filename).with_suffix('.yaml')

# If the file doesn't exist fallback to the default settings file
if not settings_file_path.exists():
    settings_file_path = Path(settings_directory, 'default').with_suffix('.yaml')

# Read settings file
with open(settings_file_path, 'r') as settings_file:
    settings = load(settings_file, Loader=Loader)

# Add script name to settings, so it's added to the log
settings['script'] = os.path.basename(sys.argv[0])
settings['settings_file'] = str(settings_file_path)

# Create convenient variables
cfg = settings["DeepSwarm"]
nodes = settings["Nodes"]
left_cost_is_better = operator.le if cfg['metrics'] == 'loss' else operator.ge
