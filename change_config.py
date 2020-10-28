import json
import os
import argparse


def change_item(k, v, task, model, out_dir):
    file_dir = os.path.join(out_dir, task)
    file = os.path.join(file_dir, "config.json")
    with open(file, 'r') as f:
        d = json.load(f)
    with open(file, 'w') as f:
        d[k] = v
        json.dump(d, f)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--key",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--value",
    default=None,
    type=float,
    required=True,
)

parser.add_argument(
    "--task",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--model",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--out_dir",
    default="/home/",
    type=str,
    required=False,
)

args = parser.parse_args()

change_item(args.key, args.value, args.task, args.model, args.out_dir)
