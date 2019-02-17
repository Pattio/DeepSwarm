# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import os
import yaml
from pathlib import Path


project_path = Path(os.path.dirname(os.path.dirname(__file__)))
with open(project_path / 'settings.yaml', 'r') as settings_file:
    settings = yaml.load(settings_file)

cfg = settings["DeepSwarm"]
nodes = settings["Nodes"]
