import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

## custom layers

class instanceNormalization(tfkl.Layer):
    ''' Custom layer that implements instance norm '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        super().build(input_shape)
    def call(self, x):
        return tfkl.Lambda(lambda t: tf.contrib.layers.instance_norm(t))(x)

class oneMinusLayer(tfkl.Layer):
    ''' Custom layer that implements `f(x) = 1 - x`. Aids with tfjs conversion.'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        super().build(input_shape)
    def call(self, x):
        return tfkl.Lambda(lambda t: 1-t)(x)


def conv_block(x, filters=32, kernel_size=2, strides=1, padding="SAME", norm=True, lrelu=0,):
    conv_params = {'filters': filters, 'kernel_size': kernel_size, 'strides': strides, 'padding': padding}
    x = tfkl.Conv2D(**conv_params)(x)
    if norm:
        x = instanceNormalization()(x)
    if isinstance(lrelu, float) and lrelu >= 0:
        x = tfkl.LeakyReLU(alpha=lrelu)(x)
    return x


def deconv_block(x, filters=32, kernel_size=2, strides=1, padding="SAME", norm=True, lrelu=0):
    conv_params = {'filters': filters, 'kernel_size': kernel_size, 'strides': strides, 'padding': padding}
    x = tfkl.Conv2DTranspose(**conv_params)(x)
    if norm:
        x = instanceNormalization()(x)
    if isinstance(lrelu, float) and lrelu >= 0:
        x = tfkl.LeakyReLU(alpha=lrelu)(x)
    return x


def resnet_block(net):
    x = net
    #x = tfkl.ZeroPadding2D([1, 1])(x)
    x = conv_block(x, int(x.shape[-1]), 3, 1, norm=False)
    #x = tfkl.ZeroPadding2D([1, 1])(x)
    x = conv_block(x, int(x.shape[-1]), 3, 1, norm=False, lrelu=False)
    net = tfkl.Add()([x, net])
    net = instanceNormalization()(net)
    net = tfkl.LeakyReLU(alpha=0)(net)
    return net


def generator(input_shape=(64, 64, 3), n_resnet_blocks=1):
    generator_input = tfkl.Input(input_shape)

    net = generator_input
    # encode
    net = conv_block(net, filters=32, kernel_size=7, strides=1)
    net = conv_block(net, filters=64, kernel_size=3, strides=2)
    net = conv_block(net, filters=128, kernel_size=3, strides=2)
    # transform
    for _ in range(n_resnet_blocks):
        net = resnet_block(net)
    # decode
    net = deconv_block(net, filters=64, kernel_size=3, strides=2)
    net = deconv_block(net, filters=32, kernel_size=3, strides=2)
    net = conv_block(net, filters=input_shape[-1], kernel_size=7, strides=1, norm=False)

    return tfk.models.Model(generator_input, net)


def discriminator(input_shape=(64, 64, 3)):
    discriminator_input = tfkl.Input(input_shape)

    net = discriminator_input
    net = conv_block(net,  64, 4, 2, norm=True, lrelu=0.2)
    net = conv_block(net, 128, 4, 2, norm=True, lrelu=0.2)
    net = conv_block(net, 256, 4, 2, norm=True, lrelu=0.2)
    net = conv_block(net, 512, 4, 2, norm=True, lrelu=0.2)
    net = conv_block(net, 1, 4, 2, norm=False, lrelu=False, padding="VALID")
    net = tfkl.Activation('sigmoid')(tfkl.Flatten()(net))

    return tfk.models.Model(discriminator_input, net)
