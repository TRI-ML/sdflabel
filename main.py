import os
import sys
import argparse
import configparser
import shutil

import utils.data as config
from pipelines.train_css import train_css
from pipelines.refine_css import refine_css
from pipelines.refine_css_demo import refine_css_demo
from pipelines.evaluate_dump import evaluate

import torch
import numpy
seed = 1
numpy.random.seed(seed)
torch.manual_seed(seed)


def main():
    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='config.ini', help='config file')
    parser.add_argument('--train', '-trn', action='store_true', help='train css network')
    parser.add_argument('--refine', '-ref', action='store_true', help='pose/shape refinement')
    parser.add_argument('--evaluate', '-eval', action='store_true', help='evaluate generated dumps')
    parser.add_argument('--demo', '-d', action='store_true', help='demo refinement')

    # Parse arguments
    args = parser.parse_args()

    # Read config file
    cfg = args.config
    cfgparser = configparser.ConfigParser()
    res = cfgparser.read(cfg)
    if len(res) == 0:
        print("Error: None of the config files could be read")
        sys.exit(1)

    # Create a log folder if needed:
    log_dir = config.read_cfg_string(cfgparser, 'log', 'dir', default='log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Store config
    shutil.copyfile(cfg, os.path.join(log_dir, os.path.basename(log_dir) + '.ini'))

    # Execution
    if args.train:
        train_css(cfgparser)  # Train CSS net
    elif args.refine:
        refine_css(cfgparser)  # Refine shapes and poses
    elif args.evaluate:
        evaluate(cfgparser)  # Evaluate generated dump
    elif args.demo:
        refine_css_demo(cfgparser)  # Demo refinement


if __name__ == '__main__':
    main()
