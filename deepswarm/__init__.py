# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import os
import operator
import sys
from yaml import load, Loader
from pathlib import Path

# Get name of script which started execution
script_name = os.path.basename(sys.argv[0])
# Retrieve name without the extension
filename = os.path.splitext(script_name)[0]
# Locate settings directory
settings_directory = Path(os.path.dirname(os.path.dirname(__file__))) / 'settings'
# Create settings file path corresponding to the script name
settings_file_path = Path(settings_directory, filename).with_suffix('.yaml')
# If file doesn't exist fallback to default settings file
if not settings_file_path.exists():
    settings_file_path = Path(settings_directory, 'default').with_suffix('.yaml')
# Read settings file
with open(settings_file_path, 'r') as settings_file:
    settings = load(settings_file, Loader=Loader)
# Add script name to settings, so it's added to the log
settings['script_name'] = script_name
# Create convenient variables
cfg = settings["DeepSwarm"]
nodes = settings["Nodes"]
left_cost_is_better = operator.le if cfg['metrics'] == 'loss' else operator.ge
