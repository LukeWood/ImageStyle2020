"""
Convert input model to a coreml model.

Depends on https://github.com/ImageStyle/coremltools (added support for
Instance Normalization)
"""
from argparse import ArgumentParser
import os

CONV_FILTERS = [32, 64, 128]
NUM_RESIDS = 5

def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--model', type=str,
                        dest='model', help='model path',
                        metavar='MODEL', required=True)
    parser.add_argument('--width', type=int,
                        dest='width', help='width of model input',
                        metavar='WIDTH', required=True)
    parser.add_argument('--height', type=int,
                        dest='height', help='height of model input',
                        metavar='HEIGHT', required=True)
    parser.add_argument('--output', type=str,
                        dest='output', help='output path',
                        metavar='OUTPUT', required=True)
    parser.add_argument('--conv-filters', type=int, nargs='+',
                        dest='conv_filters',
                        help='number of filters in conv layers in transform net',
                        metavar='CONV_FILTERS', default=CONV_FILTERS)
    parser.add_argument('--num-resids', type=int,
                        dest='num_resids',
                        help='number of residual blocks in transform net',
                        metavar='NUM_RESIDS', default=NUM_RESIDS)
    return parser

def check_opts(opts):
    assert os.path.exists(opts.model), "model path doesn't exist"

parser = build_parser()
options = parser.parse_args()
check_opts(options)

import coremltools
from keras.preprocessing import image
from keras.models import load_model, Model
from keras.layers import Input

from loss import create_loss_fn
from transform import TransformNet


def convert_multiarray_output_to_image(spec, feature_name, is_bgr=False):
    """
    Convert an output multiarray to be represented as an image
    This will modify the Model_pb spec passed in.
    Example:
        model = coremltools.models.MLModel('MyNeuralNetwork.mlmodel')
        spec = model.get_spec()
        convert_multiarray_output_to_image(spec,'imageOutput',is_bgr=False)
        newModel = coremltools.models.MLModel(spec)
        newModel.save('MyNeuralNetworkWithImageOutput.mlmodel')
    Parameters
    ----------
    spec: Model_pb
        The specification containing the output feature to convert
    feature_name: str
        The name of the multiarray output feature you want to convert
    is_bgr: boolean
        If multiarray has 3 channels, set to True for RGB pixel order or false for BGR
    """
    for output in spec.description.output:
        if output.name != feature_name:
            continue
        if output.type.WhichOneof('Type') != 'multiArrayType':
            raise ValueError("%s is not a multiarray type" % output.name)
        array_shape = tuple(output.type.multiArrayType.shape)
        channels, height, width = array_shape
        from coremltools.proto import FeatureTypes_pb2 as ft
        if channels == 1:
            output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
        elif channels == 3:
            if is_bgr:
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('BGR')
            else:
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('RGB')
        else:
            raise ValueError("Channel Value %d not supported for image inputs" % channels)
        output.type.imageType.width = width
        output.type.imageType.height = height



inputs = Input(shape=(options.width, options.height, 3))
transform_net = TransformNet(inputs, options.conv_filters, options.num_resids)
model = Model(inputs=inputs, outputs=transform_net)
model.load_weights(options.model)

model = coremltools.converters.keras.convert(model, input_names='inputImage',
                                                    image_input_names='inputImage',
                                                    output_names='outputImage')
spec = model.get_spec()
convert_multiarray_output_to_image(spec, 'outputImage', is_bgr=False)
img_out_model = coremltools.models.MLModel(spec)
img_out_model.save(options.output)
