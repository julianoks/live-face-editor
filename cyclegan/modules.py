import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

class conv_block(tfkl.Layer):
    def __init__(self, filters=32, kernel_size=2, strides=1, padding="VALID",
        norm=True, lrelu=0, **kwargs):
        self.conv_params = {'filters': filters, 'kernel_size': kernel_size, 'strides': strides, 'padding': padding}
        self.norm = norm
        self.lrelu = lrelu
        super(conv_block, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.fns = []
        self.fns.append(tfkl.Conv2D(**self.conv_params))
        if self.norm:
            self.fns.append(tfkl.BatchNormalization(virtual_batch_size=1))
        if isinstance(self.lrelu, float) and self.lrelu >= 0:
            self.fns.append(tfkl.LeakyReLU(alpha=self.lrelu))
        super(conv_block, self).build(input_shape)
    
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
    
    def build(self, input_shape):
        self.fns = []
        self.fns.append(tfkl.Conv2DTranspose(**self.conv_params))
        if self.norm:
            self.fns.append(tfkl.BatchNormalization(virtual_batch_size=1))
        if isinstance(self.lrelu, float) and self.lrelu >= 0:
            self.fns.append(tfkl.LeakyReLU(alpha=self.lrelu))
        super(deconv_block, self).build(input_shape)
    
    def call(self, x):
        for fn in self.fns:
            x = fn(x)
        return x



class resnet_block(tfkl.Layer):
    def __init__(self, **kwargs):
        super(resnet_block, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.fns = [
            tfkl.ZeroPadding2D([1, 1]),
            conv_block(int(input_shape[-1]), 3, 1),
            tfkl.ZeroPadding2D([1, 1]),
            conv_block(int(input_shape[-1]), 3, 1, lrelu=False),
        ]
        super(resnet_block, self).build(input_shape)
    
    def call(self, x):
        net = x
        for fn in self.fns:
            net = fn(net)
        return tfkl.LeakyReLU(alpha=0)(tfkl.Add()([x, net]))


def generator(n_blocks=5, input_shape=(64, 64, 3)):
    return tfk.models.Sequential([
        tfkl.ZeroPadding2D([3, 3], input_shape=input_shape),
        conv_block(32, 7, 1, "VALID"),
        conv_block(64, 7, 2, "SAME"),
        conv_block(128, 7, 2, "SAME"),
        *[resnet_block() for _ in range(n_blocks)],
        deconv_block(64, 4, 2, "SAME"),
        deconv_block(32, 4, 2, "SAME"),
        deconv_block(3, 7, 1, "SAME", lrelu=False),
        tfkl.Activation('tanh')
    ])

def discriminator(input_shape=(64, 64, 3)):
    return tfk.models.Sequential([
        conv_block(64, 4, 2, "SAME", norm=False, lrelu=0.2, input_shape=input_shape),
        conv_block(128, 4, 2, "SAME", lrelu=0.2),
        conv_block(256, 4, 2, "SAME", lrelu=0.2),
        conv_block(512, 4, 2, "SAME", lrelu=0.2),
        conv_block(512, 2, 2, "VALID", lrelu=0.2),
        conv_block(1, 2, 1, "VALID", norm=False, lrelu=False),
        tfkl.Flatten()
    ])
