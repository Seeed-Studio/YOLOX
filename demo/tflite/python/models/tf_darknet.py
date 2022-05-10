from tensorflow import keras
from .tf_network_blocks import TFBaseConv, TFCSPLayer, TFDWConv, TFFcous, TFSPPBottleneck


class TFCSPDarknet(keras.layers.Layer):
    def __init__(self,
                 dep_mul,
                 wid_mul,
                 out_features=("dark3", "dark4", "dark5"),
                 depthwise=False,
                 act="silu",
                 name=None,
                 w=None):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = TFDWConv if depthwise else TFBaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        self.n = [f'{name}.stem.conv',
                  [f'{name}.dark2.0', f'{name}.dark2.1'],
                  [f'{name}.dark3.0', f'{name}.dark3.1'],
                  [f'{name}.dark4.0', f'{name}.dark4.1'],
                  [f'{name}.dark5.0', f'{name}.dark5.1', f'{name}.dark5.2'],
                  ]

        # stem
        self.stem = TFFcous(3, base_channels, 3, act=act, name=self.n[0], w=w)

        # dark2
        self.dark2 = keras.Sequential([
            Conv(base_channels, base_channels * 2, 3, 2, act=act, name=self.n[1][0], w=w),
            TFCSPLayer(base_channels * 2,
                       base_channels * 2,
                       n=base_depth,
                       depthwise=depthwise,
                       act=act,
                       name=self.n[1][1],
                       w=w)
            ])

        # dark3
        self.dark3 = keras.Sequential([
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act, name=self.n[2][0], w=w),
            TFCSPLayer(base_channels * 4,
                       base_channels * 4,
                       n=base_depth * 3,
                       depthwise=depthwise,
                       act=act,
                       name=self.n[2][1],
                       w=w)
            ])

        # dark4
        self.dark4 = keras.Sequential([
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act, name=self.n[3][0], w=w),
            TFCSPLayer(base_channels * 8,
                       base_channels * 8,
                       n=base_depth * 3,
                       depthwise=depthwise,
                       act=act,
                       name=self.n[3][1],
                       w=w)
            ])

        # dark5
        self.dark5 = keras.Sequential([
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act, name=self.n[4][0], w=w),
            TFSPPBottleneck(base_channels * 16, base_channels * 16, act=act, name=self.n[4][1], w=w),
            TFCSPLayer(base_channels * 16,
                       base_channels * 16,
                       n=base_depth,
                       depthwise=depthwise,
                       shortcut=False,
                       act=act,
                       name=self.n[4][2],
                       w=w)
            ])

    def call(self, inputs):
        outputs = {}
        x = self.stem(inputs)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
