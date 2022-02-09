import logging

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

logger = logging.getLogger(__name__)

# select strategy if gpus in use
strategy = tf.distribute.MirroredStrategy()


# Did not use dataclass due to weird __dict__ behaviour
class Layer:
    """
    A wrapper for the key information to be passed to a given Layer object from tf.
    """

    def __init__(
        self,
        kind=Dense,
        units=10,
        activation="relu",
        kernel_regularizer="l2",
        rate=0.0,
    ):
        """
        Initalizer for the layer wrapper. Passes the kind of layer to be used (tf class),
        as well as the key parameters to be passed.

        :param kind: tf.keras.Layer, kind of layer to initialize with the other parameters
        :param units: int, number of nodes for layer
        :param activation: str, keras shorthand of the activation function to be used
        :param kernel_regularizer: str, either 'l1', 'l2', or None
        :param rate: float, dropout rate, only applies to Dropout layer
        """

        # store attributes
        self.kind = kind
        self.units = units
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer
        self.rate = rate

        # filter out irrelevant parameters for Dropout and BatchNormalization layers
        if self.kind in (Dropout, BatchNormalization):
            self.units = None
            self.activation = None
            self.kernel_regularizer = None
            if self.kind is BatchNormalization:
                self.rate = None
        else:
            self.rate = None

    def get_attr(self):
        """
        Function that returns all attributes as a dict, except for the filtered ones and the kind of layer.

        :return: dict, all relevant parameters to be passed to tf.keras.Layer
        """
        d = (
            self.__dict__.copy()
        )  # copy the dict, so that 'kind' attr is not deleted
        del d[
            "kind"
        ]  # remove kind from dict, as it isn't passed to tf.keras.Layer
        return {
            k: v for k, v in d.items() if v is not None
        }  # remove all parameters that were filtered previously


def get_layers(shape, activation, regularizer, drop_out, batch_norm):
    """
    Helper function to create all the relevant layers, simplification of general structure.

    :param shape: array-like of int, number of nodes for each hidden layer
    :param activation: str, activation function shorthand, to be used for all layers
    :param regularizer: str, regularizer to be used for all layers
    :param drop_out: float, drop-out rate to be applied between all layers
    :param batch_norm: bool, if batch normalization is to be applied between all layers
    :return: array-like of Layer, list containing the individual Layer objects to be used for ann construction
    """
    layer_list = []  # initialize list to fill layers with

    # shape includes node counts for all layers but drop-out and batch_normalization
    for i in shape:
        # apply dense layer with parameters
        layer_list.append(
            Layer(
                Dense,
                units=i,
                activation=activation,
                kernel_regularizer=regularizer,
            )
        )

        # add in drop-out, does nothing if drop_out=0.0
        layer_list.append(Layer(Dropout, rate=drop_out))
        if (
            batch_norm
        ):  # if batch normalization is to be applied, apply between all layers
            layer_list.append(Layer(BatchNormalization))
    return layer_list


def ann(layer_list, out_shape, loss, opt):
    """
    Function that creates the relevant sequential model and compiles it.

    :param layer_list: array-like of Layer, generated using get_layers
    :param out_shape: int, number of nodes in the output layer
    :param loss: str, loss function shorthand
    :param opt: tf.keras.Optimizer, to be used to optimize the ann
    :return: tf.keras.Sequential, compiled model with the given parameters
    """
    tf.random.set_seed(1)  # set seed for consistency of results
    model = tf.keras.Sequential()  # initialize empty model
    for layer in layer_list:  # add layers in one by one
        model.add(
            layer.kind(**layer.get_attr())
        )  # unpack all attributes but the kind of layer
    model.add(
        Dense(units=out_shape, activation="linear")
    )  # use linear activation for the output layer
    model.compile(
        opt, loss=loss, metrics=["mae"]
    )  # compile using the relevant loss, opt, and 'mae' as our KPM
    return model


def fit_ann(
    x_tr,
    y_tr,
    layer_list,
    rate=0.0015,
    loss="mae",
    epochs=3_000,
    validation_split=0.15,
    batch_size=1_000,
    verbose=2,
):
    """
    Function to generate and fit an ann with the given parameters and data.

    :param x_tr: 2d-array, training input data, events along 1st axis, features along 2nd axis
    :param y_tr: 1d- or 2d-array, training target data, events along 1st axis
    :param layer_list: array-like of Layer, generated using get_layers
    :param rate: float, learning rate
    :param loss: str, loss function shorthand
    :param epochs: int, number of epochs to use for fitting
    :param validation_split: float, percentage of data to use for validation
    :param batch_size: int, size of batch to use for training
    :return: tuple of tf.keras.Model and Model.history objects
    """

    # in case of more than one target to fit per event
    if len(y_tr.shape) > 1:
        out_sh = y_tr.shape[1]
    else:
        out_sh = 1

    # use Adagrad optimizer as it has shown the best results
    opt = tf.keras.optimizers.Adagrad(learning_rate=rate)
    est = ann(layer_list, out_sh, loss, opt)  # generate estimator

    # fit estimator and store history
    hist = est.fit(
        x_tr,
        y_tr,
        batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_split=validation_split,
    )
    return est, hist
