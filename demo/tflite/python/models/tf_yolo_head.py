from tensorflow import keras
import tensorflow as tf
from .tf_network_blocks import TFBaseConv, TFDWConv, TFConv2d


class TFYOLOXHead(keras.layers.Layer):
    def __init__(self,
                 num_classes,
                 width=1.0,
                 strides=[8, 16, 32],
                 in_channels=[256, 512, 1024],
                 act="silu",
                 depthwise=False,
                 name="head",
                 w=None):
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs, self.reg_convs, self.cls_preds, \
        self.reg_preds, self.obj_preds, self.stems = [], [], [], [], [], []
        Conv = TFDWConv if depthwise else TFBaseConv
        self.n = [f'{name}.cls_convs',
                  f'{name}.reg_convs',
                  f'{name}.cls_preds',
                  f'{name}.reg_preds',
                  f'{name}.obj_preds',
                  f'{name}.stems']

        for i in range(len(in_channels)):
            self.stems.append(
                TFBaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                    name=f'{self.n[5]}.{i}',
                    w=w
                )
            )
            self.cls_convs.append(
                keras.Sequential([
                    Conv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                        name=f'{self.n[0]}.{i}.0',
                        w=w
                    ),
                    Conv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                        name=f'{self.n[0]}.{i}.1',
                        w=w
                    )
                ])
            )
            self.reg_convs.append(
                keras.Sequential([
                    Conv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                        name=f'{self.n[1]}.{i}.0',
                        w=w
                    ),
                    Conv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                        name=f'{self.n[1]}.{i}.1',
                        w=w
                    )
                ])
            )
            self.cls_preds.append(
                TFConv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    ksize=1,
                    stride=1,
                    name=f'{self.n[2]}.{i}',
                    w=w
                )
            )
            self.reg_preds.append(
                TFConv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    ksize=1,
                    stride=1,
                    name=f'{self.n[3]}.{i}',
                    w=w
                )
            )
            self.obj_preds.append(
                TFConv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    ksize=1,
                    stride=1,
                    name=f'{self.n[4]}.{i}',
                    w=w
                )
            )

        self.strides = strides
        # self.grids = [tf.zeros(1)] * len(in_channels)

    def call(self, inputs):
        outputs = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, inputs)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = tf.concat(
                [reg_output, tf.sigmoid(obj_output), tf.sigmoid(cls_output)], -1
            )
            outputs.append(output)

        self.hw = [x.shape[1:3] for x in outputs]
        outputs = tf.concat(
            [tf.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[3]]) for x in outputs], 1
        )
        if self.decode_in_inference:
            return self.decode_outputs(outputs, dtype=tf.float32)
        else:
            return outputs

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = tf.meshgrid(tf.range(hsize), tf.range(wsize), indexing='ij')
            grid = tf.reshape(tf.stack((xv, yv), 2), [1, -1, 2])
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(tf.fill((*shape, 1), stride))

        grids = tf.cast(tf.concat(grids, axis=1), dtype)
        strides = tf.cast(tf.concat(strides, axis=1), dtype)

        outputs_xy = (outputs[..., :2] + grids) * strides
        outputs_wh = tf.exp(outputs[..., 2:4]) * strides
        outputs = tf.concat([outputs_xy, outputs_wh, outputs[..., 4:]], -1)
        return outputs