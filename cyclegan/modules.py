import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

def build_propagate_trainable_vars(build_fn):
    ''' assumes all layers are in the list `self.fns` '''
    def new_build_fn(self, input_shape):
        build_fn(self, input_shape)
        self.call(tfkl.Input(input_shape[1:]))
        for f in self.fns:
            self.trainable_weights.extend(f.trainable_weights)
        super(self.__class__, self).build(input_shape)
    return new_build_fn

def instance_normalization():
    return tfkl.Lambda(lambda x: tf.contrib.layers.instance_norm(x))

class conv_block(tfkl.Layer):
    def __init__(self, filters=32, kernel_size=2, strides=1, padding="VALID",
        norm=True, lrelu=0, **kwargs):
        self.conv_params = {'filters': filters, 'kernel_size': kernel_size, 'strides': strides, 'padding': padding}
        self.norm = norm
        self.lrelu = lrelu
        super(conv_block, self).__init__(**kwargs)
    
    @build_propagate_trainable_vars
    def build(self, input_shape):
        self.fns = []
        self.fns.append(tfkl.Conv2D(**self.conv_params))
        if self.norm:
            self.fns.append(instance_normalization())
        if isinstance(self.lrelu, float) and self.lrelu >= 0:
            self.fns.append(tfkl.LeakyReLU(alpha=self.lrelu))

    def call(self, x):
        for fn in self.fns:
            x = fn(x)
        return x


class deconv_block(tfkl.Layer):
    def __init__(self, filters=32, kernel_size=2, strides=1, padding="VALID",
        norm=True, lrelu=0, **kwargs):
        self.conv_params = {'filters': filters, 'kernel_size': kernel_size, 'strides': strides, 'padding': padding}
        self.norm = norm
        self.lrelu = lrelu
        super(deconv_block, self).__init__(**kwargs)
    
    @build_propagate_trainable_vars
    def build(self, input_shape):
        self.fns = []
        self.fns.append(tfkl.Conv2DTranspose(**self.conv_params))
        if self.norm:
            self.fns.append(instance_normalization())
        if isinstance(self.lrelu, float) and self.lrelu >= 0:
            self.fns.append(tfkl.LeakyReLU(alpha=self.lrelu))
    
    def call(self, x):
        for fn in self.fns:
            x = fn(x)
        return x



class resnet_block(tfkl.Layer):
    def __init__(self, **kwargs):
        super(resnet_block, self).__init__(**kwargs)
    
    @build_propagate_trainable_vars
    def build(self, input_shape):
        self.fns = [
            tfkl.ZeroPadding2D([1, 1]),
            conv_block(int(input_shape[-1]), 3, 1),
            tfkl.ZeroPadding2D([1, 1]),
            conv_block(int(input_shape[-1]), 3, 1, lrelu=False),
        ]
    
    def call(self, x):
        net = x
        for fn in self.fns:
            net = fn(net)
        return tfkl.LeakyReLU(alpha=0)(tfkl.Add()([x, net]))


def generator(n_blocks=1, input_shape=(64, 64, 3)):
    trunk_fns = [
        conv_block(16, 2, 2, "VALID", input_shape=input_shape),
        conv_block(32, 2, 2, "SAME"),
        conv_block(32, 1, 1, "SAME"),
        *[resnet_block() for _ in range(n_blocks)],
        deconv_block(16, 4, 2, "SAME", norm=True),
        deconv_block(8, 4, 2, "SAME"),
        conv_block(4, 1, 1, "SAME", lrelu=False)
    ]

    generator_input = tfkl.Input(input_shape)
    trunk = generator_input
    for fn in trunk_fns:
        trunk = fn(trunk)

    mask = trunk
    mask = resnet_block()(mask)
    mask = conv_block(4, 1, 1, "SAME")(mask)
    mask = conv_block(1, 1, 1, "SAME", lrelu=False)(mask)
    mask = tfkl.Activation('sigmoid')(mask)

    image = trunk
    image = resnet_block()(image)
    image = conv_block(3, 1, 1, "SAME", norm=False)(image)
    image = conv_block(3, 1, 1, "SAME", lrelu=False, norm=False)(image)
    image = tfkl.Activation('sigmoid')(image)

    output = tfkl.Lambda(lambda ins: 
        (ins[0] * ins[1]) + ((1 - ins[0]) * ins[2]))(
            [mask, generator_input, image])
    
    return tfk.models.Model(generator_input, output)


def discriminator(input_shape=(64, 64, 3)):
    return tfk.models.Sequential([
        conv_block(32, 2, 2, "SAME", norm=False, lrelu=0.2, input_shape=input_shape),
        conv_block(64, 2, 2, "SAME", lrelu=0.2),
        conv_block(64, 2, 1, "SAME", lrelu=0.2),
        resnet_block(),
        resnet_block(),
        conv_block(1, 1, 1, "SAME", lrelu=0.2),
        tfkl.Flatten(),
        tfkl.Dense(64),
        tfkl.LeakyReLU(alpha=0.2),
        tfkl.Dense(1),
        tfkl.Activation('sigmoid')
    ])
