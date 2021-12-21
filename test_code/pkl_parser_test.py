import argparse
import pickle
from pathlib import Path
import sys
sys.path.append(Path(__file__).parent.parent.as_posix())
from recorder.actor_tree import ActorTree
from param import *
from utils.label_types import ObjectLabel
from utils.transform import *


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--pkl',
        help='Path to pkl file form root path'
    )
    args = argparser.parse_args()

    obj_labels = None
    with open("{}/{}".format(ROOT_PATH, args.pkl), 'rb') as pkl_file:
        obj_labels = pickle.load(pkl_file)

    for obj_label in obj_labels:
        print(obj_label)


if __name__ == "__main__":
    main()
