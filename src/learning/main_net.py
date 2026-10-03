"""Command-line entry point for network training and inference."""

import argparse
import os
import sys

# Support direct execution from the repository root.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

from learning.inference import infer
from learning.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Lepid-IO network pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train the network")
    p_train.add_argument("--config", default="configs/lepid_io.json")
    p_train.add_argument("--device", default=None)

    p_infer = sub.add_parser("infer", help="run batched CSV inference")
    p_infer.add_argument("--config", default="configs/lepid_io.json")
    p_infer.add_argument("--ckpt", required=True)
    p_infer.add_argument("--csv_dir", default=None,
                         help="override the CSV directory configured by --split")
    p_infer.add_argument("--split", choices=("train", "val", "test"),
                         default="test")
    p_infer.add_argument("--output", default="results/net_output.csv")
    p_infer.add_argument("--device", default=None)
    p_infer.add_argument("--batch_size", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        train(args.config, args.device)
    elif args.command == "infer":
        infer(args.config, args.ckpt, args.csv_dir, args.output,
              args.device, args.batch_size, args.split)


if __name__ == "__main__":
    main()
