import os
import numpy as np
import tensorflow.keras as tfk
import tensorflow.keras.backend as tfkb
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose, Concatenate

EPSILON = 1e-8
CONCRETE_TEMPERATURE = 2/3

class BVAE():
    '''
    Based on https://github.com/EmilienDupont/vae-concrete. Thank you!
    '''
    def __init__(self, beta=1, latent_cont_dim=2,
                 latent_disc_dim=2, hidden_dim=128, filters=(64, 64, 64),
                 learning_rate=1e-3, num_epochs=5, batch_size=32, val_split=0.1):
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
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.val_split = val_split

    def fit(self, x_train):
        self.input_shape = tuple(x_train.shape[1:])
        self._build_models()
        self.model.fit(x_train, x_train,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            validation_split=self.val_split)
        return self
    
    def save(self, foldername='saved_model'):
        os.mkdir(foldername)
        toPath = lambda fname: os.path.join(foldername, fname)
        model = self.modelGenerateWithTranslation
        with open(toPath('keras.json'), 'w') as json_file:
            json_file.write(model.to_json())
        model.save_weights(toPath('keras.h5'))



    def _build_models(self):
        # Encoder
        inputs = Input(shape=self.input_shape)

        Q_0 = Conv2D(self.input_shape[2], (2, 2), padding='same',
                     activation='relu')
        Q_1 = Conv2D(self.filters[0], (2, 2), padding='same', strides=(2, 2),
                     activation='relu')
        Q_2 = Conv2D(self.filters[1], (3, 3), padding='same', strides=(1, 1),
                     activation='relu')
        Q_3 = Conv2D(self.filters[2], (3, 3), padding='same', strides=(1, 1),
                     activation='relu')
        Q_4 = Flatten()
        Q_5 = Dense(self.hidden_dim, activation='relu')
        Q_funcs = (Q_0, Q_1, Q_2, Q_3, Q_4, Q_5)
        # Latent
        Q_z_mean = Dense(self.latent_cont_dim)
        Q_z_log_var = Dense(self.latent_cont_dim)
        Q_c = Dense(self.latent_disc_dim, activation='softmax')
        # Decoder
        out_shape = (int(self.input_shape[0] / 2), int(self.input_shape[1] / 2), self.filters[2])
        G_0 = Dense(self.hidden_dim, activation='relu')
        G_1 = Dense(int(np.prod(out_shape)), activation='relu')
        G_2 = Reshape(out_shape)
        G_3 = Conv2DTranspose(self.filters[2], (3, 3), padding='same',
                              strides=(1, 1), activation='relu')
        G_4 = Conv2DTranspose(self.filters[1], (3, 3), padding='same',
                              strides=(1, 1), activation='relu')
        G_5 = Conv2DTranspose(self.filters[0], (2, 2), padding='valid',
                              strides=(2, 2), activation='relu')
        G_6 = Conv2D(self.input_shape[2], (2, 2), padding='same',
                     strides=(1, 1), activation='sigmoid')
        G_funcs = (G_0, G_1, G_2, G_3, G_4, G_5, G_6)

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
        x = tfkb.flatten(x)
        x_generated = tfkb.flatten(x_generated)
        reconstruction_loss = self.input_shape[0] * self.input_shape[1] * tfkb.binary_crossentropy(x, x_generated)
        
        kl_normal_loss = tfkb.mean(0.5 * (tfkb.sum(tfkb.square(self.z_mean) + tfkb.exp(self.z_log_var) - 1 - self.z_log_var, axis=1)))
        
        alpha_neg_entropy = tfkb.sum(self.alpha * tfkb.log(self.alpha + EPSILON), axis=1)
        kl_disc_loss = tfkb.log(float(self.alpha.get_shape().as_list()[1])) + tfkb.mean(alpha_neg_entropy - tfkb.sum(self.alpha, axis=1))

        return reconstruction_loss + (self.beta * (kl_normal_loss + kl_disc_loss))

    def _sampling_normal(self, args):
        z_mean, z_log_var = args
        shape = tfkb.shape(z_mean)
        epsilon = tfkb.random_normal(shape=shape, mean=0., stddev=1.)
        return z_mean + tfkb.exp(z_log_var / 2) * epsilon

    def _sampling_concrete(self, alpha):
        shape = tfkb.shape(alpha)
        uniform = tfkb.random_uniform(shape=shape)
        gumbel = -1 * tfkb.log(-1 * tfkb.log(uniform + EPSILON) + EPSILON)
        logit = (tfkb.log(alpha + EPSILON) + gumbel) / CONCRETE_TEMPERATURE
        return tfkb.softmax(logit)


if __name__ == "__main__":
    model = BVAE()
    model.fit(np.random.randn(72,64,64,3))
    model.save()