import modules

import numpy as np
import tensorflow as tf
import tensorflowjs

import json
import os
from collections import OrderedDict


class gan(object):
    default_hyperparams = OrderedDict([
        ('tensorboard_dir', 'tensorboard'),
        ('saved_models_dir', 'saved_models'),

        ('image_shape', '[64, 64, 3]'),

        ('n_epochs', 10),
        ('batch_size', 16),

        ('disc_steps_per_gen_step', 3),
        ('disc_learning_rate', 1e-4),
        ('gen_learning_rate', 1e-4),
        ('generator_n_resnet_blocks', 1),

        ('lambda_1', 1),
        ('lambda_2', 0.1),
        ('lambda_3', 0.01),

        ('tensorboard_summary_period', 100),
        ('tfjs_saving_period', 1000),
        ('tf_config_proto', '{"log_device_placement": false}'),
        ('argparse_args', {}),
    ])


    def __init__(self, **kwargs):
        # hyperparams
        self.__dict__.update(self.default_hyperparams)
        self.__dict__.update(kwargs)
        # deserialize some hyperparams
        for k in ('image_shape', 'tf_config_proto'):
            self.__dict__[k] = json.loads(self.__dict__[k])
        assert isinstance(self.image_shape, list) and len(self.image_shape)==3
        self.tf_config = tf.ConfigProto(**self.tf_config_proto)

        # instantiate optimizers
        self.disc_optimizer = tf.train.AdamOptimizer(self.disc_learning_rate)
        self.gen_optimizer = tf.train.AdamOptimizer(self.gen_learning_rate)

        # instantiate models
        self.discriminator_A = modules.discriminator(input_shape=self.image_shape)
        self.discriminator_B = modules.discriminator(input_shape=self.image_shape)
        self.generator_A2B = modules.generator(input_shape=self.image_shape,
            n_resnet_blocks=self.generator_n_resnet_blocks)
        self.generator_B2A = modules.generator(input_shape=self.image_shape,
            n_resnet_blocks=self.generator_n_resnet_blocks)
        for model in (self.discriminator_A, self.discriminator_B, self.generator_A2B, self.generator_B2A):
            # 'sgd' and 'mse' are not actually used, just placeholders for compiling
            model.compile('sgd', 'mse')
    

    def discriminator_loss(self, real_A, real_B):
        MSE = lambda a,b: tf.reduce_mean((a-b)**2)
        # model outputs
        fake_A = self.generator_B2A(real_B)
        fake_B = self.generator_A2B(real_A)
        fooled_A = self.discriminator_A(fake_A)
        fooled_B = self.discriminator_B(fake_B)
        not_fooled_A = self.discriminator_A(real_A)
        not_fooled_B = self.discriminator_B(real_B)

        # log some images
        tf.summary.image("A", real_A, max_outputs=1)
        tf.summary.image("A2B(A)", fake_B, max_outputs=1)
        tf.summary.image("B", real_B, max_outputs=1)
        tf.summary.image("B2A(B)", fake_A, max_outputs=1)

        # calculate losses
        fooled_A = MSE(fooled_A, tf.zeros_like(fooled_A))
        fooled_B = MSE(fooled_B, tf.zeros_like(fooled_B))
        not_fooled_A = MSE(not_fooled_A, tf.ones_like(not_fooled_A))
        not_fooled_B = MSE(not_fooled_B, tf.ones_like(not_fooled_B))

        # aggregate loss with standard sum
        return fooled_A + fooled_B + not_fooled_A + not_fooled_B


    def generator_loss(self, real_A, real_B):
        MSE = lambda a,b: tf.reduce_mean((a-b)**2)
        MAE = lambda a,b: tf.reduce_mean(tf.abs(a-b))

        # model outputs
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
        return tf.reduce_mean(
            (self.lambda_1 * (fooled_A + fooled_B))
            + (self.lambda_2 * (iden_A + iden_B))
            + (self.lambda_3 * (recon_A + recon_B)))


    def fit(self, fnames, classes):
        try: tf.gfile.DeleteRecursively(self.tensorboard_dir)
        except: pass
        tboard_writer = tf.summary.FileWriter(self.tensorboard_dir)

        dataset = self.get_dataset(fnames, classes).make_initializable_iterator()
        As, Bs = dataset.get_next()

        gen_step = self.fit_generator_step(As, Bs)
        disc_step = self.fit_discriminator_step(As, Bs)

        with tf.Session(config=self.tf_config) as sess:
            sess.run([dataset.initializer, tf.global_variables_initializer()])
            # tboard_writer.add_graph(sess.graph)

            for epoch in range(self.n_epochs):
                for step in range(len(fnames) // self.batch_size):
                    do_gen_step = step%(self.disc_steps_per_gen_step + 1) == 0
                    train_step = gen_step if do_gen_step else disc_step
                    
                    if self.tensorboard_summary_period and (step % self.tensorboard_summary_period) == 0:
                        summary, _ = sess.run([tf.summary.merge_all(), train_step])
                        tboard_writer.add_summary(summary, step)
                    else:
                        sess.run([train_step])

                    if self.tfjs_saving_period and (step % self.tfjs_saving_period) == 0:
                        self.save()
                    
            self.save()
        
        return self
    

    def fit_generator_step(self, As, Bs):
        g_loss = self.generator_loss(As, Bs)
        A2B_vars = self.generator_A2B.trainable_variables
        B2A_vars = self.generator_B2A.trainable_variables
        tf.summary.scalar('generator_loss', g_loss)
        return self.gen_optimizer.minimize(g_loss, var_list=A2B_vars+B2A_vars)
    

    def fit_discriminator_step(self, As, Bs):
        d_loss = self.discriminator_loss(As, Bs)
        A_vars = self.discriminator_A.trainable_variables
        B_vars = self.discriminator_B.trainable_variables
        tf.summary.scalar('discriminator_loss', d_loss)
        return self.disc_optimizer.minimize(d_loss, var_list=A_vars+B_vars)


    def get_dataset(self, fnames, classes):
        ''' Gets dataset.
        Args:
            fnames: a list of image filenames
            classes: a list of classes. `fnames[classes == 1]` are assigned class "A"
        
        Returns:
            A tensorflow dataset
        '''
        fnames = np.array(fnames)
        classes = np.array(classes)
        assert len(classes) == len(fnames)


        def load_image(file):
            image = tf.image.decode_jpeg(tf.read_file(file), channels=3)
            image = tf.expand_dims(image, axis=0)
            image = tf.image.resize_nearest_neighbor(image, self.image_shape[:2])
            image = tf.divide(tf.cast(image[0], 'float32'), 255.)
            return image

        As = (tf.data.Dataset.from_tensor_slices(fnames[classes == 1])
                .shuffle(2000)
                .map(load_image))
        Bs = (tf.data.Dataset.from_tensor_slices(fnames[classes != 1])
                .shuffle(2000)
                .map(load_image))

        return (tf.data.Dataset.zip((As, Bs))
            .repeat()
            .batch(self.batch_size)
            .prefetch(2000))


    def save(self):
        to_path = lambda *names: os.path.join(self.saved_models_dir, *names)
        try: tf.gfile.DeleteRecursively(self.saved_models_dir)
        except: pass
        for folder in (self.saved_models_dir, to_path('A2B'), to_path('B2A')):
            tf.gfile.MkDir(folder)

        # save argparse args (for the sake of provenance)
        with tf.gfile.GFile(to_path('argparse_args.json'), 'w') as f:
            f.write(json.dumps(self.argparse_args, indent=4))

        # save to keras and tfjs
        generators = {'A2B': self.generator_A2B, 'B2A': self.generator_B2A}
        for name, model in generators.items():
            tensorflowjs.converters.save_keras_model(model, to_path(name, 'tfjs'))

