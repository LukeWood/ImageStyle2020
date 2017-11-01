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

    parser.add_argument('-p', '--pad',
                        help='add reflection padding to input image',
                        dest='pad', action='store_true')

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


def pad(img, border_size):
    return np.pad(img, ((border_size, border_size),
                        (border_size, border_size),
                        (0,0)), mode='reflect')


def unpad(img, border_size):
    return img[border_size: -border_size, border_size: -border_size]


parser = build_parser()
options = parser.parse_args()
check_opts(options)

from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from scipy.misc import imsave

from transform import TransformNet

# Get input image
input_img = image.load_img(options.input)
input_img = image.img_to_array(input_img)
if options.pad:
    input_img = pad(input_img, options.border_size)
input_img = np.expand_dims(input_img, axis=0)


# Load model
_, h, w, c = input_img.shape
inputs = Input(shape=(h, w, c))

# TODO: determine conv_filters and num_resids from weights file?
transform_net = TransformNet(inputs, options.conv_filters, options.num_resids)

model = Model(inputs, transform_net)
model.load_weights(options.model)

output_img = model.predict([input_img])[0]
if options.pad:
    output_img = unpad(output_img, options.border_size)
imsave(options.output, output_img)
