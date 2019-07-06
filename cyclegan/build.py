import os
import argparse
import pandas as pd
import tensorflow as tf
from gan import gan

def run(celeba_dir='', feature='Black_Hair', **gan_kwargs):
    fnames = os.path.join(celeba_dir, 'img_align_celeba', '*.jpg')
    fnames = tf.gfile.Glob(fnames)
    fnames = pd.Series(fnames, [n.rsplit('/', 1)[1] for n in fnames])
    fnames = fnames.sort_index().values

    classes = os.path.join(celeba_dir, 'list_attr_celeba.csv')
    with tf.gfile.Open(classes) as f:
        classes = pd.read_csv(f).set_index('image_id')
        assert feature in list(classes), '`feature` must be one of `{}`, got `{}`'.format(list(classes), feature)
        classes = classes[feature].sort_index().values

    gan(**gan_kwargs).fit(fnames, classes)


def build_argparser():
    parser = argparse.ArgumentParser(description='Build the CycleGAN models.')

    parser.add_argument('--feature', default='Black_Hair', help='Feature to train on')

    to_path = lambda *args: os.path.join(os.path.abspath(os.path.dirname(__file__)), *args)

    parser.add_argument('--celeba_dir', default=to_path('..', 'celeba-dataset'), help='Path to the celeba dataset')
    parser.add_argument('--tensorboard_dir', default=to_path('tensorboard'), help='Path to save tensorboard files in')
    parser.add_argument('--saved_models_dir', default=to_path('saved_models'), help='Path to save models in')

    for k, v in gan.default_hyperparams.items():
        if k not in ('tensorboard_dir', 'saved_models_dir'):
            parser.add_argument('--{}'.format(k), default=v, type=type(v), help='Default value = `{}`'.format(v))

    return parser
    

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(**args.__dict__)
