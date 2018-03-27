from math import sqrt
import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback


def put_kernels_on_grid(kernel, pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''

    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1:
                    raise ValueError("Please use a composite number of filters")
                return i, int(n / i)

    (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))  # 3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))  # 3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype=tf.uint8)


class CustomTensorboard(Callback):
    def __init__(self, log_dir, validation_generator=None):
        super(CustomTensorboard, self).__init__()
        self.log_dir = log_dir
        self.summary_functions = []
        self.validation_generator = validation_generator

    def add_summary(self, model_to_summary):
        self.summary_functions.append(model_to_summary)
        return self

    def set_model(self, model):
        self.model = model
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()

        self.merged = tf.summary.merge(
            [f(model) for f in self.summary_functions])
        self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)

    def on_epoch_begin(self, epoch, logs=None):
        sess = K.get_session()
        (inputs, targets) = self.validation_generator.next()
        feed_dict = {self.model.inputs[0]: inputs}
        self.writer.add_summary(sess.run(self.merged, feed_dict=feed_dict), epoch)


def convKernelSummary(model):
    layers = model.layers
    summaries = []
    for layer in layers:
        if isinstance(layer, keras.layers.convolutional.Conv2D):
            nChannels = layer.kernel.get_shape()[2]
            if nChannels == 3 or nChannels == 1:
                summaries.append(tf.summary.image(
                    layer.kernel.name, put_kernels_on_grid(layer.kernel)))
            else:
                padded_kernels = put_kernels_on_grid(layer.kernel)
                for channel in range(nChannels):
                    summaries.append(tf.summary.image(
                        layer.kernel.name + "/{}".format(channel),
                        tf.expand_dims(padded_kernels[:, :, :, channel], -1)))
    return tf.summary.merge(summaries)


def convActivationSummary(model):
    layers = model.layers
    summaries = []
    input_node = model.inputs[0]
    summaries.append(tf.summary.image("activations/" + input_node.name, tf.expand_dims(input_node[1, :, :, :], 0)))

    for layer in layers:
        if isinstance(layer, keras.layers.convolutional.Conv2D):
            activations = tf.expand_dims(layer.output[1, :, :, :], -2)
            padded_activations = put_kernels_on_grid(activations)
            summaries.append(tf.summary.image("activations_" + layer.output.name, padded_activations))

    return tf.summary.merge(summaries)
