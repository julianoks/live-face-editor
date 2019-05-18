import os
import subprocess
import numpy as np
import tensorflow as tf
import modules

class gan(object):
    def __init__(self):
        self.n_epochs = 10
        self.batch_size = 16
        self.gen_steps_per_iteration = 3
        self.discriminator_A = modules.discriminator()
        self.discriminator_B = modules.discriminator()
        self.generator_A2B = modules.generator()
        self.generator_B2A = modules.generator()
    
    def discriminator_loss(self, real_A, real_B):
        MSE = lambda a,b: tf.reduce_mean((a-b)**2)
        # calculate values
        fake_A = self.generator_B2A(real_B)
        fake_B = self.generator_A2B(real_A)
        fooled_A = self.discriminator_A(fake_A)
        fooled_B = self.discriminator_B(fake_B)
        unfooled_A = self.discriminator_A(real_A)
        unfooled_B = self.discriminator_B(real_B)
        # calculate losses
        fooled_A = MSE(fooled_A, tf.zeros_like(fooled_A))
        fooled_B = MSE(fooled_B, tf.zeros_like(fooled_B))
        unfooled_A = MSE(unfooled_A, tf.ones_like(unfooled_A))
        unfooled_B = MSE(unfooled_B, tf.ones_like(unfooled_B))
        # aggregate loss with standard sum
        return fooled_A + fooled_B + unfooled_A + unfooled_B

    def generator_loss(self, real_A, real_B):
        MSE = lambda a,b: tf.reduce_mean((a-b)**2)
        MAE = lambda a,b: tf.reduce_mean(tf.abs(a-b))
        # calculate values
        fake_A = self.generator_B2A(real_B)
        fake_B = self.generator_A2B(real_A)
        fooled_A = self.discriminator_A(fake_A)
        fooled_B = self.discriminator_B(fake_B)
        recon_A = self.generator_B2A(fake_B)
        recon_B = self.generator_A2B(fake_A)
        iden_A = self.generator_B2A(real_A)
        iden_B = self.generator_A2B(real_B)
        # calculate losses
        fooled_A = MSE(fooled_A, tf.ones_like(fooled_A))
        fooled_B = MSE(fooled_A, tf.ones_like(fooled_B))
        iden_A = MAE(iden_A, real_A)
        iden_B = MAE(iden_B, real_B)
        recon_A = MAE(recon_A, real_A)
        recon_B = MAE(recon_B, real_B)
        # aggregate loss with weighted sum
        return tf.reduce_mean(fooled_A + fooled_B + iden_A + iden_B + (10 * (recon_A + recon_B)))

    def fit(self, fnames, classes):
        # TODO: visualize progress with tensorboard
        dataset = self.get_dataset(fnames, classes).make_initializable_iterator()
        As, Bs = dataset.get_next()
        gen_step = self.fit_generator_step(As, Bs)
        disc_step = self.fit_discriminator_step(As, Bs)
        with tf.Session() as sess:
            sess.run([dataset.initializer, tf.global_variables_initializer()])
            for epoch in range(self.n_epochs):
                for step in range(len(fnames) // self.batch_size):
                    print("Iteration {} / {}".format(step, len(fnames) // self.batch_size))
                    sess.run([gen_step, disc_step])
                self.save()
        return self
    
    def fit_generator_step(self, As, Bs):
        g_loss = self.generator_loss(As, Bs)
        A2B_vars = self.generator_A2B.trainable_variables
        B2A_vars = self.generator_B2A.trainable_variables
        # TODO: track trainable weights using standard Keras API
        A2B_vars = sum(sum([[b._trainable_weights for b in l2] for l2 in [l1.fns if 'fns' in l1.__dict__ else [] for l1 in self.generator_A2B.layers]], []), [])
        B2A_vars = sum(sum([[b._trainable_weights for b in l2] for l2 in [l1.fns if 'fns' in l1.__dict__ else [] for l1 in self.generator_B2A.layers]], []), [])
        return tf.train.AdamOptimizer().minimize(g_loss, var_list=A2B_vars+B2A_vars)
    
    def fit_discriminator_step(self, As, Bs):
        d_loss = self.discriminator_loss(As, Bs)
        A_vars = self.discriminator_A.trainable_variables
        B_vars = self.discriminator_B.trainable_variables
        # TODO: track trainable weights using standard Keras API
        A_vars = sum(sum([[b._trainable_weights for b in l2] for l2 in [l1.fns if 'fns' in l1.__dict__ else [] for l1 in self.discriminator_A.layers]], []), [])
        B_vars = sum(sum([[b._trainable_weights for b in l2] for l2 in [l1.fns if 'fns' in l1.__dict__ else [] for l1 in self.discriminator_B.layers]], []), [])
        return tf.train.AdamOptimizer().minimize(d_loss, var_list=A_vars+B_vars)

    def get_dataset(self, fnames, classes):
        def load_image(file):
            image = tf.image.decode_jpeg(tf.read_file(file), channels=3)
            image = tf.expand_dims(image, axis=0)
            image = tf.image.resize_nearest_neighbor(image, [64, 64])
            image = tf.divide(tf.cast(image[0], 'float32'), 255.)
            return image
        fnames = np.array(fnames)
        As = (tf.data.Dataset.from_tensor_slices(fnames[classes == 1])
                .shuffle(2000)
                .map(load_image))
        Bs = (tf.data.Dataset.from_tensor_slices(fnames[classes != 1])
                .shuffle(2000)
                .map(load_image))
        return tf.data.Dataset.zip((As, Bs)).batch(self.batch_size).prefetch(2000)

    def save(self, foldername='saved_models'):
        this_dir_path = os.path.abspath(os.path.dirname(__file__))
        foldername = os.path.join(this_dir_path, foldername)
        subprocess.run(['rm', '-r', foldername])
        os.mkdir(foldername)
        toPath = lambda *names: os.path.join(foldername, *names)
        os.mkdir(toPath('A2B'))
        os.mkdir(toPath('B2A'))
        # save to keras and tfjs
        generators = {'A2B': self.generator_A2B, 'B2A': self.generator_B2A}
        for name, model in generators.items():
            model.save(toPath(name, 'keras_model.h5'))
            subprocess.run(['tensorflowjs_converter', '--input_format=keras', \
                toPath(name, 'keras_model.h5'), toPath(name, 'tfjs')])
        return self
