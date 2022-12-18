#!/usr/bin/python3
from pathlib import Path
import sys
import os
# Project Root Path
ROOT_PATH = Path(__file__).parent.as_posix()
print("Project Root PATH: {}".format(ROOT_PATH))

RAW_DATA_PATH = "{}/{}".format(ROOT_PATH, 'raw_data')
DATASET_PATH = "{}/{}".format(ROOT_PATH, 'dataset')
