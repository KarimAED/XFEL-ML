import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

strategy = tf.distribute.MirroredStrategy()


# Did not use dataclass due to weird __dict__ behaviour
class Layer:

    def __init__(self, kind=Dense, units=10, activation="relu", kernel_regularizer="l2", rate=0.0):
        self.kind = kind
        self.units = units
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer
        self.rate = rate

        if self.kind in (Dropout, BatchNormalization):
            self.units = None
            self.activation = None
            self.kernel_regularizer = None
            if self.kind is BatchNormalization:
                self.rate = None
        else:
            self.rate = None

    def get_attr(self):
        d = self.__dict__.copy()
        del d["kind"]
        return {k: v for k, v in d.items() if v is not None}


def get_layers(shape, activation, regularizer, drop_out, batch_norm):
    layer_list = []
    for i in shape:
        layer_list.append(Layer(Dense, units=i, activation=activation, kernel_regularizer=regularizer))
        layer_list.append(Layer(Dropout, rate=drop_out))
        if batch_norm:
            layer_list.append(Layer(BatchNormalization))
    return layer_list


def ann(layer_list, out_shape, loss, opt):
    tf.random.set_seed(1)
    model = tf.keras.Sequential()
    for layer in layer_list:
        model.add(layer.kind(**layer.get_attr()))
    model.add(Dense(units=out_shape, activation="linear"))
    model.compile(opt, loss=loss, metrics=["mae"])
    return model


def fit_ann(x_tr, y_tr, layer_list, rate=0.0015, loss="mae", epochs=3_000, validation_split=0.15, batch_size=1_000):

    if len(y_tr.shape) > 1:
        out_sh = y_tr.shape[1]
    else:
        out_sh = 1

    opt = tf.keras.optimizers.Adagrad(learning_rate=rate)
    est = ann(layer_list, out_sh, loss, opt)
    hist = est.fit(x_tr, y_tr, batch_size,
                   epochs=epochs, verbose=2,
                   validation_split=validation_split)
    return est, hist
