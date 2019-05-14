import os
import subprocess
import numpy as np
import tensorflow as tf
tfk = tf.keras
import modules

class gan(object):
    def __init__(self):
        self.optimizer = tfk.optimizers.Adam(0.0002, 0.5)
        self.n_epochs = 5
        self.batch_size = 16
        self.gen_steps_per_iteration = 3
        self.build()
    
    def build(self):
        self.discriminator_A = modules.discriminator()
        self.discriminator_B = modules.discriminator()
        self.generator_A2B = modules.generator()
        self.generator_B2A = modules.generator()

        real_A = tfk.layers.Input(shape=[64, 64, 3])
        real_B = tfk.layers.Input(shape=[64, 64, 3])
        fake_A = self.generator_B2A(real_B)
        fake_B = self.generator_A2B(real_A)
        recon_A = self.generator_B2A(fake_B)
        recon_B = self.generator_A2B(fake_A)
        iden_A = self.generator_B2A(real_A)
        iden_B = self.generator_A2B(real_B)

        fooled_A = self.discriminator_A(fake_A)
        fooled_B = self.discriminator_B(fake_B)
        caught_A = self.discriminator_A(real_A)
        caught_B = self.discriminator_B(real_B)

        images = tfk.layers.Input(shape=[64, 64, 3])
        self.generator_models = {
            'A2B': tfk.Model(images, self.generator_A2B(images)),
            'B2A': tfk.Model(images, self.generator_B2A(images)),
        }

        self.train_discriminators = tfk.Model(inputs=(real_A, real_B), \
            outputs=(fooled_A, fooled_B, caught_A, caught_B))
        self.train_discriminators.compile(loss='mse', \
            optimizer=self.optimizer, metrics=['accuracy'])

        self.discriminator_A.trainable = False
        self.discriminator_B.trainable = False
        self.train_generators = tfk.Model(inputs=(real_A, real_B), \
            outputs=(fooled_A, fooled_B, recon_A, recon_B, iden_A, iden_B))
        self.train_generators.compile(\
            loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'], \
            loss_weights=[1, 1, 10, 10, 1, 1], optimizer=self.optimizer)


    def fit(self, fnames, classes):
        dataset = self.get_dataset(fnames, classes)
        for epoch in range(self.n_epochs):
            for step in range(len(fnames) // self.batch_size):
                self.train_discriminators.train_on_batch(dataset)
                for _ in range(self.gen_steps_per_iteration):
                    self.train_generators.train_on_batch(dataset)
                print("Iteration", step, "/", len(fnames) // self.batch_size)
                self.save()
        return self

    def get_dataset(self, fnames, classes):
        def load_image(file):
            image = tf.image.decode_jpeg(tf.read_file(file), channels=3)
            image = tf.expand_dims(image, axis=0)
            image = tf.image.resize_nearest_neighbor(image, [64, 64])
            image = tf.divide(tf.cast(image[0], 'float32'), 255.)
            return image
        fnames = np.array(fnames)
        As = tf.data.FixedLengthRecordDataset(tf.constant(fnames[classes == 1]), 11000)
        As = As.map(load_image).shuffle(2000)
        Bs = tf.data.FixedLengthRecordDataset(tf.constant(fnames[classes != 1]), 11000)
        Bs = Bs.map(load_image).shuffle(2000)
        return tf.data.Dataset.zip((As, Bs)).batch(self.batch_size)

    def save(self, foldername='saved_model'):
        this_dir_path = os.path.abspath(os.path.dirname(__file__))
        foldername = os.path.join(this_dir_path, foldername)
        subprocess.run(['rm', '-r', foldername])
        os.mkdir(foldername)
        toPath = lambda *names: os.path.join(foldername, *names)
        os.mkdir(toPath('A2B'))
        os.mkdir(toPath('B2A'))
        # save to keras
        for name in ('A2B', 'B2A'):
            self.generator_models[name].save(toPath(name, 'keras_model.h5'))
            subprocess.run(['tensorflowjs_converter', '--input_format=keras', \
                toPath(name, 'keras_model.h5'), toPath(name, 'tfjs')])
        return self
