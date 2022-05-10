import tensorflow as tf
from tensorflow import keras


class TFAct(keras.layers.Layer):
    """activation function from torch to tf"""
    def __init__(self, name=None):
        super().__init__()
        if name == "silu":
            self.act = lambda x: keras.activations.swish(x)
        elif name == "relu":
            self.act = lambda x: keras.activations.relu(x)
        elif name == "lrelu":
            self.act = lambda x: keras.activations.relu(x, alpha=0.1)
        else:
            raise AttributeError("Unsupported act type: {}".format(name))

    def call(self, inputs):
        return self.act(inputs)


class TFBN(keras.layers.Layer):
    """TensorFlow BatchNormalization wrapper"""
    def __init__(self, w=None, name=None):
        super().__init__()
        self.n = [f'{name}.bn.bias',
                  f'{name}.bn.weight',
                  f'{name}.bn.running_mean',
                  f'{name}.bn.running_var']
        self.bn = keras.layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(w[self.n[0]].numpy()),
            gamma_initializer=keras.initializers.Constant(w[self.n[1]].numpy()),
            moving_mean_initializer=keras.initializers.Constant(w[self.n[2]].numpy()),
            moving_variance_initializer=keras.initializers.Constant(w[self.n[3]].numpy()),
            epsilon=1e-3,
            momentum=0.03)

    def call(self, inputs):
        return self.bn(inputs)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else (x // 2 for x in k)  # auto-pad
    return p


class TFPad(keras.layers.Layer):

    def __init__(self, pad):
        super().__init__()
        self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])

    def call(self, inputs):
        return tf.pad(inputs, self.pad, mode='constant', constant_values=0)


class TFBaseConv(keras.layers.Layer):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""
    def __init__(self, in_channels, out_channels, ksize, stride, padding=None,
                 g=1, bias=False, act="silu", name=None, w=None):
        super().__init__()
        self.n = [f'{name}.conv.weight']
        if g==1:
            conv = keras.layers.Conv2D(
                out_channels,
                ksize,
                stride,
                'SAME' if stride == 1 else 'VALID',
                use_bias=bias,
                kernel_initializer=keras.initializers.Constant(w[self.n[0]].permute(2, 3, 1, 0).numpy()))
        else: # g = 1, otherwise in_channels
            conv = keras.layers.DepthwiseConv2D(
                ksize,
                stride,
                'SAME' if stride == 1 else 'VALID',
                use_bias=bias,
                depthwise_initializer=keras.initializers.Constant(w[self.n[0]].permute(2, 3, 0, 1).numpy()))
        self.conv = conv if stride == 1 else keras.Sequential([TFPad(autopad(ksize, padding)), conv])
        self.bn = TFBN(w=w, name=name)
        self.act = TFAct(name=act)

    def call(self, inputs):
        return self.act(self.bn(self.conv(inputs)))

    # def fusecall(self, inputs):
    #     return self.act(self.conv(inputs))


class TFDWConv(keras.layers.Layer):
    """Depthwise Conv + Conv"""
    def __init__(self, in_channels, out_channels, ksize, stride=1,
                 act="silu", name=None, w=None):
        super().__init__()
        self.n = [f'{name}.dconv',
                  f'{name}.pconv']
        self.dconv = TFBaseConv(
            in_channels, out_channels, ksize, stride, act=act, g=in_channels, name=self.n[0], w=w
        )
        self.pconv = TFBaseConv(
            in_channels, out_channels, 1, 1, act=act, name=self.n[1], w=w
        )

    def call(self, inputs):
        return self.pconv(self.dconv(inputs))


class TFBottleneck(keras.layers.Layer):
    """standard bottleneck"""
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5,
                 depthwise=False, act="silu", name=None, w=None):
        super().__init__()
        self.n = [f'{name}.conv1',
                  f'{name}.conv2']
        hidden_channels = int(out_channels * expansion)
        Conv = TFDWConv if depthwise else TFBaseConv
        self.conv1 = TFBaseConv(in_channels, hidden_channels, 1, 1, act=act, name=self.n[0], w=w)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1, act=act, name=self.n[1], w=w)
        self.use_add = shortcut and in_channels == out_channels

    def call(self, inputs):
        x=self.conv2(self.conv1(inputs))
        if self.use_add:
            x = x + inputs
        return x


class TFResLayer(keras.layers.Layer):
    """Residual layer with `in_channels` inputs"""
    def __init__(self, in_channels, name=None, w=None):
        super().__init__()
        self.n = [f'{name}.layer1',
                  f'{name}.layer2']
        mid_channels = in_channels // 2
        self.layer1 = TFBaseConv(
            in_channels, mid_channels, 1, 1, act="lrelu", name=self.n[0], w=w
        )
        self.layer2 = TFBaseConv(
            mid_channels, in_channels, 3, 1, act="lrelu", name=self.n[1], w=w
        )

    def call(self, inputs):
        x = self.layer2(self.layer1(inputs))
        return inputs + x


class TFSPPBottleneck(keras.layers.Layer):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    def __init__(self, in_channels, out_channels, ksize=(5, 9, 13),
                 act="silu", name=None, w=None):
        super().__init__()
        self.n = [f'{name}.conv1',
                  f'{name}.conv2']
        hidden_channels = in_channels // 2
        self.conv1 = TFBaseConv(in_channels, hidden_channels, 1, 1, act=act, name=self.n[0], w=w)
        self.m = [keras.layers.MaxPool2D(pool_size=x, strides=1, padding='SAME') for x in ksize]
        conv2_channels = hidden_channels * (len(ksize) +1) # pooling channels and itself
        self.conv2 = TFBaseConv(conv2_channels, out_channels, 1, 1, act=act, name=self.n[1], w=w)

    def call(self, inputs):
        x = self.conv1(inputs)
        return self.conv2(tf.concat([x] + [m(x) for m in self.m], 3))


class TFCSPLayer(keras.layers.Layer):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5,
                 depthwise=False, act="silu", name=None, w=None): # n(int): number of bottlenecks. Defalut 1.
        super().__init__()
        self.n = [f'{name}.conv1',
                  f'{name}.conv2',
                  f'{name}.conv3',
                  [f'{name}.m.{i}' for i in range(n)],
                  ]
        hidden_channels = int(out_channels * expansion)
        self.conv1 = TFBaseConv(in_channels, hidden_channels, 1, 1, act=act, name=self.n[0], w=w)
        self.conv2 = TFBaseConv(in_channels, hidden_channels, 1, 1, act=act, name=self.n[1], w=w)
        self.conv3 = TFBaseConv(2 * hidden_channels, out_channels, 1, 1, act=act, name=self.n[2], w=w)
        self.m = keras.Sequential([
                TFBottleneck(
                            hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act, name=self.n[3][i], w=w
                             ) for i in range(n)])

    def call(self, inputs):
        x_1 = self.conv1(inputs)
        x_2 = self.conv2(inputs)
        x_1 = self.m(x_1)

        return self.conv3(tf.concat((x_1, x_2), 3))


class TFFcous(keras.layers.Layer):
    """Focus width and height information into channel space."""
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu", name=None, w=None):
        super().__init__()
        self.conv = TFBaseConv(in_channels * 4, out_channels, ksize, stride, act=act, name=name, w=w)

    def call(self, inputs): # x(b, w, h, c) -> y(b, w/2, h/2, 4c)
        return self.conv(
            tf.concat(
                [inputs[:, ::2, ::2, :], inputs[:, 1::2, ::2, :],
                 inputs[:, ::2, 1::2, :], inputs[:, 1::2, 1::2, :]
                 ], 3))


class TFUpsample(keras.layers.Layer):
    """tf version of torch.nn.Upsample()"""
    def __init__(self, scale_factor, mode):
        super().__init__()
        assert scale_factor == 2, "scale_factor must be 2"
        self.upsample = lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method=mode)

    def call(self, inputs):
        return self.upsample(inputs)


class TFConv2d(keras.layers.Layer):
    """Substitution for PyTorch nn.Conv2D"""
    def __init__(self, in_channels, out_channels, ksize, stride, padding=None,
                 bias=True, name=None, w=None):
        super().__init__()
        self.n = [f'{name}.weight',
                  f'{name}.bias']
        self.conv = keras.layers.Conv2D(
            out_channels,
            ksize,
            stride,
            'VALID',
            use_bias=bias,
            kernel_initializer=keras.initializers.Constant(w[self.n[0]].permute(2, 3, 1, 0).numpy()),
            bias_initializer=keras.initializers.Constant(w[self.n[1]].numpy()) if bias else None,
        )

    def call(self, inputs):
        return self.conv(inputs)
