from argparse import ArgumentParser
import os

CONV_FILTERS = [32, 64, 128]
NUM_RESIDS = 5

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str,
                        dest='model', help='model path',
                        metavar='MODEL', required=True)

    parser.add_argument('--input', type=str,
                        dest='input', help='input image path',
                        metavar='INPUT', required=True)

    parser.add_argument('--output', type=str,
                        dest='output', help='output image path',
                        metavar='OUTPUT', required=True)

    parser.add_argument('-b', '--border-size', type=str,
                        help='border size of reflection padding',
                        dest='border_size', default=30)

    parser.add_argument('--conv-filters', type=int, nargs='+',
                        dest='conv_filters',
                        help='number of filters in conv layers in transform net',
                        metavar='CONV_FILTERS', default=CONV_FILTERS)

    parser.add_argument('--num-resids', type=int,
                        dest='num_resids',
                        help='number of residual blocks in transform net',
                        metavar='NUM_RESIDS', default=NUM_RESIDS)


    return parser


def check_opts(options):
    assert os.path.exists(options.model), "model path not found!"
    assert os.path.exists(options.input), "input path not found!"


parser = build_parser()
options = parser.parse_args()
check_opts(options)

from keras.layers import Input, Cropping2D
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from scipy.misc import imsave

from transform import TransformNet, ReflectionPadding2D

# Get input image
input_img = image.load_img(options.input)
input_img = image.img_to_array(input_img)
input_img = np.expand_dims(input_img, axis=0)


# Load model
_, h, w, c = input_img.shape
inputs = Input(shape=(h, w, c))

# TODO: determine conv_filters and num_resids from weights file?
transform_net = TransformNet(inputs, options.conv_filters, options.num_resids)

model = Model(inputs, transform_net)
model.load_weights(options.model)

# Set padding/cropping border in layers
for layer in model.layers:
    padding = int(options.border_size)
    if isinstance(layer, ReflectionPadding2D) or isinstance(layer, Cropping2D):
        layer.padding = ((padding, padding), (padding, padding))

output_img = model.predict([input_img])[0]
imsave(options.output, output_img)
