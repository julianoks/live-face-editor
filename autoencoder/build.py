import os
import tensorflow as tf
from BVAE import BVAE

def get_celeba(new_img_size=[64,64]):
    ''' There should be 202599 images in the dataset '''
    # load and process image from file
    def load_image(file):
        image = tf.image.decode_jpeg(tf.read_file(file), channels=3)
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_nearest_neighbor(image, new_img_size)
        image = tf.math.divide(tf.cast(image[0], 'float32'), 255.)
        return image
    # get dataset
    folder_pattern = os.path.join('celeba-dataset', 'img_align_celeba', '*.jpg')
    dataset = tf.data.Dataset.list_files(folder_pattern)
    dataset = dataset.shuffle(2000).map(load_image)
    dataset = tf.data.Dataset.zip((dataset, dataset))
    return dataset

def run(**kwargs):
    BVAE().fit(get_celeba()).save()

if __name__ == "__main__":
    run(epochs=1)
