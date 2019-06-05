import os
import pandas as pd
import tensorflow as tf
from gan import gan

def run(**kwargs):
    celeba_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
        '..', 'celeba-dataset')
    fnames = os.path.join(celeba_path, 'img_align_celeba', '*.jpg')
    fnames = tf.gfile.Glob(fnames)
    classes = os.path.join(celeba_path, 'list_attr_celeba.csv')
    with tf.gfile.Open(classes) as f:
        classes = pd.read_csv(f)['Smiling'].values
    gan(**kwargs).fit(fnames, classes)

if __name__ == "__main__":
    run()
