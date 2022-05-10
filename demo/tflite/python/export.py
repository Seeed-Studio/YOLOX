"""
Export a yolox model to tensorflow formats.
You should download the weights at https://github.com/Megvii-BaseDetection/YOLOX,
and save in current folder. For example:
.\tflite
    \python
        \yolox_tiny.pth

Usage:
    $ python path/to/export.py -n yolox_nano --tsize 640 --include saved_model --device cpu
                                  yolox_tiny                       tflite               gpu
                                  yolox_s
                                  yolox_x
                                  yolox_m
                                  yolox_l
"""

import torch
import tensorflow as tf
from tensorflow import keras
import argparse
from para import Para
from loguru import logger

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
root = sys.path[0]

from models.tf_yolo_pafpn import TFYOLOPAFPN
from models.tf_yolo_head import TFYOLOXHead


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument('--include',
                        default='saved_model',
                        type=str,
                        help='saved_model, tflite')
    return parser


def parse_model(model_name, model):
    p = Para(model_name)
    depth = p.depth
    width = p.width
    num_classes = p.num_classes
    in_channels = p.in_channels
    act = p.act
    depthwise = p.depthwise

    backbone = TFYOLOPAFPN(
        depth, width, in_channels=in_channels,
        act=act, depthwise=depthwise, w=model['model']
    )
    head = TFYOLOXHead(
        num_classes, width, in_channels=in_channels,
        act=act, depthwise=depthwise, w=model['model']
    )

    return keras.Sequential([backbone, head])


def predict(inputs, model):
    x = inputs
    for m in model.layers:
        x = m(x)

    return x


def main(args):
    file = str(args.name) + ".pth"
    print(args.name, args.include)
    path = os.path.join(root, file)
    logger.info("loading torch model...")
    ckpt = torch.load(path, map_location=torch.device(args.device))

    imgsz = (args.tsize, args.tsize)
    inputs = tf.keras.Input(shape=(*imgsz, 3))
    model = parse_model(args.name, ckpt)
    outputs = predict(inputs, model)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    keras_model.summary()

    if args.include == "saved_model":
        try:
            logger.info(f'Tensorflow SavedModel: starting export...')
            f = str(path).replace('.pth', '_saved_model')
            keras_model.save(f, save_format='tf')
            logger.info(f'Tensorflow SavedModel: export success with tensorflow {tf.__version__}!')
            logger.info(f'Tensorflow SavedModel: Saved as {f}')
        except Exception as e:
            logger.info(f'Tensorflow SavedModel: export failed! {e}')

    elif args.include == "tflite":
        try:
            logger.info(f'Tensorflow Lite: starting export...')
            f = str(path).replace('.pth', '_fp16.tflite')

            converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            converter.target_spec.supported_types = [tf.float16]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            open(f, "wb").write(tflite_model)
            logger.info(f'Tensorflow Lite: export success with {tf.__version__}!')
            logger.info(f'Tensorflow Lite: Saved as {f}')
        except Exception as e:
            logger.info(f'Tensorflow Lite: export failed! {e}')


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args)
