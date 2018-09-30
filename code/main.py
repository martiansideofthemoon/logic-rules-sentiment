import yaml

import tensorflow as tf
import numpy as np
import random

from bunch import bunchify

from analysis import analysis
from config.arguments import modify_arguments, modify_config, parser
from train import train
from test import test
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    args = parser.parse_args()
    modify_arguments(args)

    # Resetting the graph and setting seeds
    tf.reset_default_graph()
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.config_file, 'r') as stream:
        config = yaml.load(stream)
        args.config = bunchify(modify_config(args, config))

    logger.info(args)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'analysis':
        analysis(args)


if __name__ == '__main__':
    main()
