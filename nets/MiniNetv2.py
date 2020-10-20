import tensorflow as tf
from tensorflow.keras import layers, regularizers


def reshape_into(inputs, input_to_copy):
    return tf.image.resize(inputs, (input_to_copy.shape[1], input_to_copy.shape[2]), method=tf.image.ResizeMethod.BILINEAR)


# convolution
def convolution(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


# Traspose convolution
def transposeConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)



# Depthwise convolution
def depthwiseConv(kernel_size, strides=1, depth_multiplier=1, dilation_rate=1, use_bias=True):
    return layers.DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=depth_multiplier,
                                  padding='same', use_bias=use_bias, kernel_regularizer=regularizers.l2(l=0.0001),
                                  dilation_rate=dilation_rate)


# Depthwise convolution
def separableConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  depthwise_regularizer=regularizers.l2(l=0.0001),
                                  pointwise_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


class DepthwiseConv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        super(DepthwiseConv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        # separableConv
        self.conv = separableConv(filters=filters, kernel_size=kernel_size, strides=strides,
                                  dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, activation=True, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.ReLU()(x)

        return x





class Conv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        super(Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = convolution(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, activation=True, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.ReLU()(x)

        return x



class Residual_SeparableConv(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0):
        super(Residual_SeparableConv, self).__init__()

        self.conv = DepthwiseConv_BN(filters, kernel_size, strides=strides, dilation_rate=dilation_rate)
        self.dropout = layers.Dropout(rate=dropout)
    def call(self, inputs, training=True):

        x = self.conv(inputs, activation=False, training=training)
        x = self.dropout(x, training=training)
        if inputs.shape == x.shape:
            x = x + inputs
        x = layers.ReLU()(x)

        return x

class Residual_SeparableConv_dil(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0):
        super(Residual_SeparableConv_dil, self).__init__()
        self.conv1 = depthwiseConv(kernel_size, strides=strides, depth_multiplier=1, dilation_rate=1, use_bias=False)
        self.conv2 = depthwiseConv(kernel_size, strides=strides, depth_multiplier=1, dilation_rate=dilation_rate, use_bias=False)
        self.bn1 = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)
        self.bn2 = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)


        self.conv = convolution(filters=filters, kernel_size=1, strides=1, dilation_rate=1)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

        self.dropout = layers.Dropout(rate=dropout)
    def call(self, inputs, training=True):

        x1 = self.conv1(inputs)
        x1 = self.bn1(x1, training=training)
        x2 = self.conv2(inputs)
        x2 = self.bn2(x2, training=training)
        x1 = layers.ReLU()(x1)
        x2 = layers.ReLU()(x2)

        x = self.conv(x1 + x2)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)

        if inputs.shape == x.shape:
            x = x + inputs

        x = layers.ReLU()(x)

        return x


class MininetV2Module(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0):
        super(MininetV2Module, self).__init__()

        self.conv1 = Residual_SeparableConv(filters, kernel_size, strides=strides, dilation_rate=1, dropout=dropout)
        self.conv2 = Residual_SeparableConv(filters, kernel_size, strides=1, dilation_rate=dilation_rate, dropout=dropout)


    def call(self, inputs, training=True):

        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        return x


class MininetV2Module_dil2(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0):
        super(MininetV2Module_dil2, self).__init__()

        self.conv1 = Residual_SeparableConv_dil(filters, kernel_size, strides=strides, dilation_rate=dilation_rate, dropout=dropout)
        self.conv2 = Residual_SeparableConv_dil(filters, kernel_size, strides=1, dilation_rate=dilation_rate*2, dropout=dropout)


    def call(self, inputs, training=True):

        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        return x


class MininetV2Module_dil(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, dropout=0):
        super(MininetV2Module_dil, self).__init__()

        self.conv1 = Residual_SeparableConv_dil(filters, kernel_size, strides=strides, dilation_rate=1, dropout=dropout)
        self.conv2 = Residual_SeparableConv_dil(filters, kernel_size, strides=1, dilation_rate=dilation_rate, dropout=dropout)


    def call(self, inputs, training=True):

        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        return x


class MininetV2Downsample(tf.keras.Model):
    def __init__(self, filters, depthwise=True):
        super(MininetV2Downsample, self).__init__()
        if depthwise:
            self.conv = DepthwiseConv_BN(filters, kernel_size=3, dilation_rate=1, strides=2)
        else:
            self.conv = Conv_BN(filters, kernel_size=3, dilation_rate=1, strides=2)

    def call(self, inputs, training=True):

        x = self.conv(inputs, training=training)

        return x


class MininetV2Upsample(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1):
        super(MininetV2Upsample, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = transposeConv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, last=False, training=True):
        x = self.conv(inputs)
        if not last:
            x = self.bn(x, training=training)
            x = layers.ReLU()(x)

        return x



class MiniNetv2(tf.keras.Model):

    def __init__(self, num_classes, **kwargs):
        super(MiniNetv2, self).__init__(**kwargs)
        self.down_b = MininetV2Downsample(32, depthwise=False)
        self.down_b2 = MininetV2Downsample(128, depthwise=True)

        self.down1 = MininetV2Downsample(16, depthwise=False)
        self.conv_mod_0 = MininetV2Module(32, 3, strides=1, dilation_rate=1)

        self.down2 = MininetV2Downsample(64, depthwise=True)
        self.conv_mod_1 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_2 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_3 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_4 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_44 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.down3 = MininetV2Downsample(128, depthwise=True)
        self.conv_mod_5 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_6 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=4)
        self.conv_mod_7 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=8)
        self.conv_mod_8 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=16)
        self.conv_mod_9 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=1)
        self.conv_mod_10 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=2)
        self.conv_mod_11 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=4)
        self.conv_mod_12 = MininetV2Module_dil(128, 3, strides=1, dilation_rate=8)
        self.conv_mod_17 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.conv_mod_19 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.classify = convolution(num_classes, 1, strides=1, dilation_rate=1, use_bias=True)


    def call(self, inputs, training=True):
        x1 = self.down1(inputs, training=training)
        x2 = self.down2(x1, training=training)
        x = self.conv_mod_1(x2, training=training)
        x = self.conv_mod_2(x, training=training)
        x = self.conv_mod_3(x, training=training)
        x = self.conv_mod_4(x, training=training)
        xx = self.conv_mod_44(x, training=training)
        x = self.down3(xx, training=training)
        x = self.conv_mod_5(x, training=training)
        x = self.conv_mod_6(x, training=training)
        x = self.conv_mod_7(x, training=training)
        x = self.conv_mod_8(x, training=training)
        x = self.conv_mod_9(x, training=training)
        x = self.conv_mod_10(x, training=training)
        x = self.conv_mod_11(x, training=training)
        x = self.conv_mod_12(x, training=training)
        x = reshape_into(x, x2)
        aux1 = self.down_b2(self.down_b(inputs, training=training))
        x = x + aux1

        x = self.conv_mod_17(x, training=training)
        x = reshape_into(x, x1)
        x = self.conv_mod_19(x, training=training)

        x = self.classify(x, training=training)

        x = reshape_into(x, inputs)

        x = tf.keras.activations.softmax(x, axis=-1)
        return x

 
