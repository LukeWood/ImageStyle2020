from loss import create_loss_fn
import imageio
from PIL import Image
from util import count_num_samples
from transform import TransformNet
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Input
from keras.callbacks import Callback
from keras.models import Model
from argparse import ArgumentParser
import os

tf.compat.v1.disable_eager_execution()

CONTENT_WEIGHT = 15.
STYLE_WEIGHT = 100.
TV_WEIGHT = 2e2
BATCH_SIZE = 4
NUM_EPOCHS = 2
TRAIN_PATH = 'data'
CONV_FILTERS = [32, 64, 128]
NUM_RESIDS = 5


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--load-model-from', type=str, dest='load_model_from',
                        help='checkpoint to load model from',
                        metavar='LOAD_MODEL_FROM', required=False, default=False)

    parser.add_argument('--model-output', type=str,
                        dest='model_output', help='model output path',
                        metavar='MODEL_OUTPUT', required=True)

    parser.add_argument('--model-input', type=str,
                        dest='model_input',
                        help='path to model to train (if continuing training)',
                        metavar='MODEL_INPUT', required=False)

    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint', help='checkpoint directory',
                        metavar='CHECKPOINT', default=False)

    parser.add_argument('--checkpoint-freq', type=int, dest='checkpoint_freq',
                        help='# of batches to checkpoint after',
                        default='100', metavar='CHECKPOINT_FREQ', required=False)

    parser.add_argument('--test', type=str,
                        dest='test', help='test image path',
                        metavar='TEST', default=False)

    parser.add_argument('--test-dir', type=str,
                        dest='test_dir', help='test image save dir',
                        metavar='TEST_DIR', default=False)

    parser.add_argument('--test-increment', type=int,
                        dest='test_increment', help='number of batches to test after',
                        metavar='TEST_INCREMENT', default=100)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--steps-per-epoch', type=int,
                        dest='steps_per_epoch',
                        help='number of batches of samples per epoch, ' +
                             '(should be # of samples / batch size)',
                        metavar='BATCH_SIZE', default=None)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

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
    assert os.path.exists(opts.style), "style image path not found!"
    assert os.path.exists(opts.train_path), "train path not found!"
    if opts.test or opts.test_dir:
        assert os.path.exists(opts.test), "test image not found!"
        assert os.path.exists(opts.test_dir), "test directory not found!"
    if opts.test:
        assert options.test_dir is not False,\
            "test output dir must be given with test"
    if opts.model_input:
        assert os.path.exists(opts.model_input), "input model path not found!"
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0


parser = build_parser()
options = parser.parse_args()
check_opts(options)


def create_gen(img_dir, target_size, batch_size):
    datagen = ImageDataGenerator()
    gen = datagen.flow_from_directory(img_dir, target_size=target_size,
                                      batch_size=batch_size, class_mode=None)

    def tuple_gen():
        for img in gen:
            if img.shape[0] != batch_size:
                continue

            # (X, y)
            # X will go through TransformNet,
            # y will go through VGG
            yield (img/255., img)
        print("Done generating tuples")
    return tuple_gen()


class OutputPreview(Callback):
    def __init__(self, test_img_path, increment, preview_dir_path):
        test_img = np.array(image.load_img(test_img_path))
        test_img = np.array(Image.fromarray(test_img).resize((256, 256)))
        test_target = image.img_to_array(test_img)
        test_target = np.expand_dims(test_target, axis=0)
        self.test_img = test_target

        self.preview_dir_path = preview_dir_path

        self.increment = increment
        self.iteration = 0

    def on_batch_end(self, batch, logs={}):
        if (self.iteration % self.increment == 0):
            output_img = model.predict(self.test_img)[0]
            fname = '%d.jpg' % self.iteration
            out_path = os.path.join(self.preview_dir_path, fname)
            imageio.imwrite(out_path, output_img)

        self.iteration += 1


style_img = image.load_img(options.style)
style_target = image.img_to_array(style_img)

inputs = Input(shape=(256, 256, 3))
transform_net = TransformNet(inputs, options.conv_filters, options.num_resids)
model = Model(inputs=inputs, outputs=transform_net)
loss_fn = create_loss_fn(style_target, options.content_weight,
                         options.style_weight, options.tv_weight,
                         options.batch_size)
model.compile(optimizer='adam', loss=loss_fn)

if options.model_input:
    model_2 = tf.keras.models.load_model(options.model_input, compile=False)
    model.set_weights(model_2.get_weights())
    print("Loaded model weights from", options.model_input)

gen = create_gen(options.train_path, target_size=(256, 256),
                 batch_size=options.batch_size)

if options.steps_per_epoch is None:
    num_samples = count_num_samples(options.train_path)
    options.steps_per_epoch = num_samples // options.batch_size

callbacks = []

if options.test:
    print("Saving sample transformations to", options.test_dir)
    callbacks.append(OutputPreview(options.test, options.test_increment,
                                   options.test_dir))
if options.checkpoint:
    print("Checkpointing to", options.checkpoint)
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        save_freq=options.checkpoint_freq,
        filepath=options.checkpoint, monitor="loss", save_best_only=True))

print("beginning to fit model")
model.fit(gen, verbose=True, steps_per_epoch=options.steps_per_epoch,
          epochs=options.epochs, callbacks=callbacks)
model.save(options.model_output)
