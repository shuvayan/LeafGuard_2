# Path setup, and access the config.yml file, datasets folder & trained models

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import sys
import os


from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from strictyaml import YAML, load

import leafguard_capstone as lf

# Project Directories
PACKAGE_ROOT = Path(lf.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yaml"

print(PACKAGE_ROOT)
print(ROOT)

DATASET_DIR = PACKAGE_ROOT / "processed_data"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "saved_models"

print(DATASET_DIR)
print(TRAINED_MODEL_DIR)

#def find_config_file() -> Path:
#    """Locate the configuration file."""
#    
#    if CONFIG_FILE_PATH.is_file():
#        return CONFIG_FILE_PATH
#    
#    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


#def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
#    """Parse YAML containing the package configuration."""

#    if not cfg_path:
#        cfg_path = find_config_file()

#    if cfg_path:
#        with open(cfg_path, "r") as conf_file:
#            parsed_config = load(conf_file.read())
#            return parsed_config
        
#    raise OSError(f"Did not find config file at path: {cfg_path}")





