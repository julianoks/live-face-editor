import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

def instance_normalization(x):
    # TODO implement instance normalization
    return x

class oneMinusLayer(tfkl.Layer):
    ''' Custom layer that implements `f(x) = 1 - x`. Aids with tfjs conversion.'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        super().build(input_shape)
    def call(self, x):
        return tfkl.Lambda(lambda t: 1-t)(x)

def conv_block(x, filters=32, kernel_size=2, strides=1, padding="VALID", norm=True, lrelu=0,):
    conv_params = {'filters': filters, 'kernel_size': kernel_size, 'strides': strides, 'padding': padding}
    x = tfkl.Conv2D(**conv_params)(x)
    if norm:
        x = instance_normalization(x)
    if isinstance(lrelu, float) and lrelu >= 0:
        x = tfkl.LeakyReLU(alpha=lrelu)(x)
    return x


def deconv_block(x, filters=32, kernel_size=2, strides=1, padding="VALID", norm=True, lrelu=0):
    conv_params = {'filters': filters, 'kernel_size': kernel_size, 'strides': strides, 'padding': padding}
    x = tfkl.Conv2DTranspose(**conv_params)(x)
    if norm:
        x = instance_normalization(x)
    if isinstance(lrelu, float) and lrelu >= 0:
        x = tfkl.LeakyReLU(alpha=lrelu)(x)
    return x


def resnet_block(net):
    x = net
    x = tfkl.ZeroPadding2D([1, 1])(x)
    x = conv_block(x, int(x.shape[-1]), 3, 1)
    x = tfkl.ZeroPadding2D([1, 1])(x)
    x = conv_block(x, int(x.shape[-1]), 3, 1, lrelu=False)
    net = tfkl.LeakyReLU(alpha=0)(tfkl.Add()([x, net]))
    return net


def generator(n_blocks=1, input_shape=(64, 64, 3)):
    generator_input = tfkl.Input(input_shape)

    trunk = generator_input
    trunk =  conv_block(trunk, 16, 2, 2, "VALID")
    trunk = conv_block(trunk, 32, 2, 2, "SAME")
    trunk = conv_block(trunk, 32, 1, 1, "SAME")
    for _ in range(n_blocks):
        trunk = resnet_block(trunk)
    trunk = deconv_block(trunk, 16, 4, 2, "SAME", norm=True)
    trunk = deconv_block(trunk, 8, 4, 2, "SAME")
    trunk = conv_block(trunk, 3, 1, 1, "SAME", lrelu=False)

    mask = trunk
    mask = resnet_block(mask)
    mask = conv_block(mask, 4, 1, 1, "SAME")
    mask = conv_block(mask, 1, 1, 1, "SAME", lrelu=False, norm=False)
    mask = tfkl.Activation('sigmoid')(mask)

    image = trunk
    image = resnet_block(image)
    image = conv_block(image, 3, 1, 1, "SAME", norm=False)
    image = conv_block(image, 3, 1, 1, "SAME", lrelu=False, norm=False)
    image = tfkl.Activation('sigmoid')(image)

    output = tfkl.Add()([
        tfkl.Multiply()([generator_input, mask]),
        tfkl.Multiply()([image, oneMinusLayer()(mask)])
    ])

    return tfk.models.Model(generator_input, output)


def discriminator(input_shape=(64, 64, 3)):
    discriminator_input = tfkl.Input(input_shape)

    net = discriminator_input
    net = conv_block(net, 32, 2, 2, "SAME", norm=False, lrelu=0.2)
    net = conv_block(net, 64, 2, 2, "SAME", lrelu=0.2)
    net = conv_block(net, 64, 2, 1, "SAME", lrelu=0.2)
    net = resnet_block(net)
    net = resnet_block(net)
    net = conv_block(net, 1, 1, 1, "SAME", lrelu=0.2)
    net = tfkl.Flatten()(net)
    net = tfkl.Dense(64)(net)
    net = tfkl.LeakyReLU(alpha=0.2)(net)
    net = tfkl.Dense(1)(net)
    net = tfkl.Activation('sigmoid')(net)

    return tfk.models.Model(discriminator_input, net)
