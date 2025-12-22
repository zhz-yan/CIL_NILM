import json
import argparse
from trainer import train
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)                   # Converting argparse Namespace to a dict.
    args.update(param)                  # Add parameters from json
    args.setdefault("repeats", 5)       #
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


# acil, bic, der, icarl, il2a
def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms.')
    parser.add_argument('--config', type=str, default='./exps/bic.json',
                        help='Json file of settings.',
                        choices=['acil', 'bic', 'der', 'icarl', 'il2a'])
    parser.add_argument('--repeats', type=int, default=1)

    return parser


if __name__ == '__main__':
    main()
