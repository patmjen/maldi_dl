import argparse
from os.path import join

import hydra
from omegaconf import OmegaConf


@hydra.main(config_path='config', config_name='config', version_base=None)
def parse_and_print_current_spectrum_config(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))


@hydra.main(config_path='config_image', config_name='config', version_base=None)
def parse_and_print_current_image_config(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))


def main(args):
    if not args.tensorboard_name:
        print('# Spectrum')
        parse_and_print_current_spectrum_config()
        print('# Image')
        parse_and_print_current_image_config()
    else:
        tensorboard_path = join('/results/tensorboard', args.tensorboard_name)
        cfg = OmegaConf.load(join(tensorboard_path, 'hparams.yaml'))
        print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tensorboard_name', default=None,  nargs='?')
    args = parser.parse_args()
    print(args)
    main(args)