import os
import glob
import pandas as pd
from gan import gan

def run(**kwargs):
    celeba_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
        '..', 'celeba-dataset')
    fnames = os.path.join(celeba_path, 'img_align_celeba', '*.jpg')
    fnames = glob.glob(fnames)
    classes = os.path.join(celeba_path, 'list_attr_celeba.csv')
    classes = pd.read_csv(classes)['Smiling'].values
    gan(**kwargs).fit(fnames, classes)

if __name__ == "__main__":
    run()
