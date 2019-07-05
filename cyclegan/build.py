import os
import pandas as pd
import tensorflow as tf
from gan import gan

def run(feature='Black_Hair', **kwargs):
    celeba_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
        '..', 'celeba-dataset')
    fnames = os.path.join(celeba_path, 'img_align_celeba', '*.jpg')
    fnames = tf.gfile.Glob(fnames)
    fnames = pd.Series(fnames, [n.rsplit('/', 1)[1] for n in fnames])
    fnames = fnames.sort_index().values

    classes = os.path.join(celeba_path, 'list_attr_celeba.csv')
    with tf.gfile.Open(classes) as f:
        classes = pd.read_csv(f).set_index('image_id')
        classes = classes[feature].sort_index().values

    gan(**kwargs).fit(fnames, classes)

if __name__ == "__main__":
    run()
