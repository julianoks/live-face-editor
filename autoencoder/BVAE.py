import os
import subprocess
import numpy as np
import tensorflow.keras as tfk
import tensorflow.keras.backend as tfkb
from tensorflow.keras.layers import Input, BatchNormalization, Dense, Lambda, LeakyReLU, Flatten, Reshape, Conv2D, Conv2DTranspose, Concatenate


class BVAE():
    def __init__(self, beta=1, latent_cont_dim=26,
                 latent_disc_dim=6, hidden_dim=64, filters=(64, 32, 16),
                 learning_rate=1e-3, epochs=1, batch_size=32,
                 examples_per_epoch=202599,
                 CONCRETE_TEMPERATURE=2/3, EPSILON=1e-8):
        self.opt = None
        self.model = None
        self.modelGenerateWithTranslation = None
        self.input_shape = None
        self.beta = beta
        self.latent_cont_dim = latent_cont_dim
        self.latent_disc_dim = latent_disc_dim
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim
        self.hidden_dim = hidden_dim
        self.filters = filters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.CONCRETE_TEMPERATURE = CONCRETE_TEMPERATURE
        self.EPSILON = EPSILON
        self.examples_per_epoch = examples_per_epoch

    def fit(self, dataset):
        self.input_shape = tuple(dataset.output_shapes[0].as_list())
        self._build_models()
        dataset = dataset.repeat(self.epochs).batch(self.batch_size)
        tensorboard_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'tensorboard')
        try: subprocess.run(['rm', '-r', tensorboard_path])
        except: pass
        tensorboard = tfk.callbacks.TensorBoard(log_dir=tensorboard_path)
        self.model.fit(dataset,
            epochs=self.epochs,
            steps_per_epoch=self.examples_per_epoch // self.batch_size,
            shuffle=True,
            callbacks=[tensorboard])
        return self
    
    def save(self, foldername='saved_model'):
        foldername = os.path.join(os.path.abspath(os.path.dirname(__file__)), foldername)
        subprocess.run(['rm', '-r', foldername])
        os.mkdir(foldername)
        toPath = lambda *names: os.path.join(foldername, *names)
        os.mkdir(toPath('tfjs'))
        model = self.modelGenerateWithTranslation
        # save to keras
        model.save(toPath('keras_model.h5'))
        # save to tfjs
        subprocess.run(['tensorflowjs_converter', '--input_format=keras', toPath('keras_model.h5'), toPath('tfjs')])



    def _build_models(self):
        # Encoder
        inputs = Input(shape=self.input_shape)

        Q_funcs = self._make_Q_funcs()
        G_funcs = self._make_G_funcs()

        Q_z_mean = Dense(self.latent_cont_dim)
        Q_z_log_var = Dense(self.latent_cont_dim)
        Q_c = Dense(self.latent_disc_dim, activation='softmax')

        # Run
        net = inputs
        for f in Q_funcs:
            net = f(net)
        self.z_mean = Q_z_mean(net)
        self.z_log_var = Q_z_log_var(net)
        self.alpha = Q_c(net)
        z = Lambda(self._sampling_normal)([self.z_mean, self.z_log_var])
        c = Lambda(self._sampling_concrete)(self.alpha)
        net = Concatenate()([z, c])
        for f in G_funcs:
            net = f(net)
        generated = net
        
        # Create models
        self.model = tfk.Model(inputs, generated)
        self.modelGenerateWithTranslation = self._createGenerateWithTranslationModel(Q_funcs, (Q_z_mean, Q_c), G_funcs)
        # Compile
        self.opt = tfk.optimizers.Adam(lr=self.learning_rate)
        self.model.compile(optimizer=self.opt, loss=self._vae_loss)
        # Loss and optimizer do not matter here as we do not train this model
        self.modelGenerateWithTranslation.compile(optimizer=self.opt, loss='mse')

    def _make_Q_funcs(self):
        return [
            Conv2D(self.input_shape[2], (2, 2), padding='same', strides=(2, 2), use_bias=True),
            LeakyReLU(alpha=0.1),
            Conv2D(self.filters[0], (2, 2), padding='same', strides=(2, 2), use_bias=False),
            BatchNormalization(), LeakyReLU(alpha=0.1),
            Conv2D(self.filters[1], (3, 3), padding='same', strides=(1, 1), use_bias=False),
            BatchNormalization(), LeakyReLU(alpha=0.1),
            Conv2D(self.filters[2], (3, 3), padding='same', strides=(1, 1), use_bias=False),
            Flatten(),
            BatchNormalization(), LeakyReLU(alpha=0.1),
            Dense(self.hidden_dim, use_bias=True),
        ]
    
    def _make_G_funcs(self):
        out_shape = (int(self.input_shape[0] / 2), int(self.input_shape[1] / 2), self.filters[2])
        return [
            Dense(self.hidden_dim, use_bias=True),
            LeakyReLU(alpha=0.1),
            Dense(int(np.prod(out_shape)), use_bias=False),
            BatchNormalization(), Reshape(out_shape),
            LeakyReLU(alpha=0.1),
            Conv2DTranspose(self.filters[2], (3, 3), padding='same', strides=(1, 1), use_bias=True),
            LeakyReLU(alpha=0.1),
            Conv2DTranspose(self.filters[1], (3, 3), padding='same', strides=(1, 1), use_bias=False),
            BatchNormalization(), LeakyReLU(alpha=0.1),
            Conv2DTranspose(self.filters[0], (2, 2), padding='valid', strides=(2, 2), use_bias=True),
            LeakyReLU(alpha=0.1),
            Conv2D(self.input_shape[2]+1, (2, 2), padding='same', strides=(1, 1), activation='sigmoid'),
        ]

    def _createGenerateWithTranslationModel(self, Q_funcs, Q_latent_funcs, G_funcs):
            inputs = Input(shape=self.input_shape)
            code_translation = Input(shape=(self.latent_dim,))
            net = inputs
            for f in Q_funcs:
                net = f(net)
            net = Concatenate()([f(net) for f in Q_latent_funcs])
            net = tfk.layers.Add()([net, code_translation])
            for f in G_funcs:
                net = f(net)
            return tfk.Model([inputs, code_translation], net)


    def _vae_loss(self, x, x_generated):
        rgb = x_generated[:,:,:, :3]
        alpha = x_generated[:,:,:, 3:]
        reconstruction_loss = tfkb.sum(alpha * ((x - rgb) ** 2))
        reconstruction_loss *= 1 - tfkb.mean(alpha)
        
        
        kl_normal_loss = tfkb.mean(0.5 * (tfkb.sum(tfkb.square(self.z_mean) + tfkb.exp(self.z_log_var) - 1 - self.z_log_var, axis=1)))
        
        alpha_neg_entropy = tfkb.sum(self.alpha * tfkb.log(self.alpha + self.EPSILON), axis=1)
        kl_disc_loss = tfkb.log(float(self.alpha.get_shape().as_list()[1])) + tfkb.mean(alpha_neg_entropy - tfkb.sum(self.alpha, axis=1))

        return reconstruction_loss + (self.beta * (kl_normal_loss + kl_disc_loss))

    def _sampling_normal(self, args):
        ''' Based on https://github.com/EmilienDupont/vae-concrete. Thank you! '''
        z_mean, z_log_var = args
        shape = tfkb.shape(z_mean)
        epsilon = tfkb.random_normal(shape=shape, mean=0., stddev=1.)
        return z_mean + tfkb.exp(z_log_var / 2) * epsilon

    def _sampling_concrete(self, alpha):
        shape = tfkb.shape(alpha)
        uniform = tfkb.random_uniform(shape=shape)
        gumbel = tfkb.log(-1 * tfkb.log(uniform + self.EPSILON) + self.EPSILON)
        logit = (tfkb.log(alpha + self.EPSILON) - gumbel) / self.CONCRETE_TEMPERATURE
        return tfkb.softmax(logit)
