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
        return image, image
    # get dataset
    folder_pattern = os.path.join(os.path.abspath(os.path.dirname(__file__)),
        'celeba-dataset', 'img_align_celeba', '*.jpg')
    dataset = tf.data.Dataset.list_files(folder_pattern)
    dataset = dataset.map(load_image)
    return dataset.shuffle(2000)

def run(**kwargs):
    BVAE(**kwargs).fit(get_celeba()).save()

if __name__ == "__main__":
    run(epochs=1)
