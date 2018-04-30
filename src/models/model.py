from keras.layers import Input, Dense
from keras.layers.merge import Add

from keras.models import Model
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import backend as K
from keras.layers.core import Lambda, Reshape
from keras.layers import Concatenate
from keras.optimizers import SGD, Adadelta
import keras
try:
    from .attention import AttentionDecoder
except:
    print("Loading theano no attention")
# from .attentionBis import Attention


def build_models(size=20, nbase=1, trainable=True, ctc_length=40, ctc=True,
                 uniform=True, input_length=None, n_output=1,
                 n_feat=4, recurrent_dropout=0, lr=0.01, res=False, attention=False, simple=False,
                 extra_output=0):
    if keras.backend.backend() == 'tensorflow':
        import tensorflow as tf

        def ctc_batch_cost(y_true, y_pred, input_length, label_length,
                           ctc_merge_repeated=False):
            '''Runs CTC loss algorithm on each batch element.

            # Arguments
                y_true: tensor (samples, max_string_length) containing the truth labels
                y_pred: tensor (samples, time_steps, num_categories) containing the prediction,
                        or output of the softmax
                input_length: tensor (samples,1) containing the sequence length for
                        each batch item in y_pred
                label_length: tensor (samples,1) containing the sequence length for
                        each batch item in y_true

            # Returns
                Tensor with shape (samples,1) containing the
                    CTC loss of each element
            '''
            label_length = tf.to_int32(tf.squeeze(label_length))
            input_length = tf.to_int32(tf.squeeze(input_length))
            sparse_labels = tf.to_int32(K.ctc_label_dense_to_sparse(y_true, label_length))

            y_pred = tf.log(tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-8)

            return tf.expand_dims(K.ctc.ctc_loss(inputs=y_pred,
                                                 labels=sparse_labels,
                                                 sequence_length=input_length,
                                                 ctc_merge_repeated=ctc_merge_repeated), 1)

    input_length = input_length
    out_layer2 = None
    inputs = Input(shape=(input_length, n_feat))
    Nbases = 4 + nbase + 1

    print("Trainable ???", trainable)

    if res:
        merge_mode = "sum"

    else:
        merge_mode = "concat"

    if simple:
        from keras.layers.recurrent import SimpleRNN as LSTM
    else:
        from keras.layers.recurrent import LSTM

    l1 = Bidirectional(LSTM(size, return_sequences=True, trainable=trainable, recurrent_dropout=recurrent_dropout),
                       merge_mode=merge_mode)(inputs)
    # if res:
    # l1 = Add()([l1, inputs])
    l2 = Bidirectional(LSTM(size, return_sequences=True, trainable=trainable, recurrent_dropout=recurrent_dropout),
                       merge_mode=merge_mode)(l1)
    if res:
        l2 = Add()([l2, l1])

    if n_output == 1:
        l3 = Bidirectional(LSTM(size, return_sequences=True, trainable=trainable, recurrent_dropout=recurrent_dropout),
                           merge_mode=merge_mode)(l2)

        if res:
            l3 = Add()([l3, l2])

        if attention:
            if input_length is not None:
                inp = 2 * size
                if res:
                    inp = size
                out_layer1 = AttentionDecoder(
                    size, Nbases, name="out_layer1", input_shape=(input_length, inp))(l3)
            else:
                out_layer1 = AttentionDecoder(size, Nbases, name="out_layer1", to_apply=True)(l3)

        else:
            out_layer1 = TimeDistributed(Dense(Nbases, activation="softmax"), name="out_layer1")(l3)

        if extra_output != 0:
            ext = []
            for n in range(extra_output):
                ext.append(TimeDistributed(
                    Dense(1, activation="sigmoid"), name="extra%i" % n)(l3))

    else:
        old = False
        if old:
            l3 = Bidirectional(LSTM(2 * size, return_sequences=True, trainable=trainable),
                               merge_mode='concat')(l2)

            if input_length != None:

                TD = TimeDistributed(Dense(Nbases, activation="softmax"), name="out_layer1")
                r = Reshape((input_length * 2, 2 * size))(l3)  # 2 * size because l3 is concat
                out_layer1 = TD(r)
            else:
                print("Latttt")

                def slice_last_d(x):

                    return x[::, ::, :2 * size]

                def slice_last_u(x):

                    return x[::, ::, 2 * size:]

                def slice_shape(input_shape):
                    shape = list(input_shape)

                    shape[-1] = 2 * size
                    return tuple(shape)

                TD = TimeDistributed(Dense(Nbases, activation="softmax"),
                                     name="out_layer1")

                l3d = Lambda(slice_last_d, slice_shape)(l3)
                l3u = Lambda(slice_last_u, slice_shape)(l3)

                out_layer1 = TD(l3d)
                out_layer2 = TD(l3u)
        else:
            l3 = Bidirectional(LSTM(size, return_sequences=True, trainable=trainable, recurrent_dropout=recurrent_dropout),
                               merge_mode=merge_mode)(l2)

            out_layer1 = TimeDistributed(
                Dense(Nbases, activation="softmax"), name="out_layer1")(l3)
            out_layer2 = TimeDistributed(
                Dense(Nbases, activation="softmax"))(l3)  # Concatenate()([l3, out_layer1]))

            if input_length is not None:

                l3b = Concatenate()([out_layer1, out_layer2])
                # 2 * size because l3 is concat
                out_layer1 = Reshape((input_length * 2, Nbases))(l3b)
                out_layer2 = None

            if extra_output != 0:
                ext = []
                for n in range(extra_output):
                    ext.append(TimeDistributed(
                        Dense(1, activation="sigmoid"), name="extra%i" % n)(l3))

    if out_layer2 is not None:
        print("ici")
        model = Model(inputs=inputs, outputs=[out_layer1, out_layer2])
    else:
        if extra_output == 0:
            model = Model(inputs=inputs, outputs=out_layer1)

        else:
            model = Model(inputs=inputs, outputs=[out_layer1] + ext)

    # ada = Adadelta(lr=lr, rho=0.95, epsilon=1e-08, decay=0.0)
    if not uniform:
        model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                      sample_weight_mode='temporal')
    else:
        model.compile(optimizer='adadelta', loss='categorical_crossentropy')

    if keras.backend.backend() != 'tensorflow':
        return model, None

    if keras.backend.backend() == 'tensorflow':

        if ctc:
            def ctc_lambda_func(args):
                y_pred, labels, input_length, label_length = args
                y_pred = y_pred[:, :, :]
                return ctc_batch_cost(labels, y_pred, input_length, label_length, ctc_merge_repeated=False)

            labels = Input(name='the_labels', shape=[ctc_length], dtype='float32')
            input_length = Input(name='input_length', shape=[1], dtype='int64')
            label_length = Input(name='label_length', shape=[1], dtype='int64')

            loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
                [out_layer1, labels, input_length, label_length])

            sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
            # rms = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0, clipvalue=0.05)
            if extra_output == 0:
                model2 = Model(inputs=[inputs, labels, input_length,
                                       label_length], outputs=loss_out)
                model2.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer="adadelta")

            else:
                def average(x):
                    x = K.mean(x, axis=-2)  # , keepdims=True)
                    return x

                def average_output_shape(input_shape):
                    shape = list(input_shape)
                    # assert len(shape) == 3  # only valid for 2D tensors

                    return tuple(shape[:-2] + [1])
                ot = []
                # inp = []
                for n in range(extra_output):
                    ot.append(Lambda(average, output_shape=average_output_shape,
                                     name="o%i" % n)(ext[n]))

                    # inp.append(Input(name='input_prop%i' % n, shape=[1], dtype='float32'))

                model2 = Model(inputs=[inputs, labels, input_length,
                                       label_length], outputs=[loss_out] + ot)

                model2.compile(loss=[lambda y_true, y_pred: y_pred] +
                               ["mean_squared_error" for i in range(extra_output)], optimizer="adadelta")

    return model, model2


import h5py
from keras.engine.topology import preprocess_weights_for_loading


def load_weights_from_hdf5_group_what_you_can(filepath, layers):

    f = h5py.File(filepath, mode='r')

    """Implements topological (order-based) weight loading.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: a list of target layers.

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file.
    """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    filtered_layers = []
    for layer in layers:
        weights = layer.weights
        if weights:
            filtered_layers.append(layer)

    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        if weight_names:
            filtered_layer_names.append(name)
    layer_names = filtered_layer_names
    print(layer_names)
    print(filtered_layers)
    """
    if len(layer_names) != len(filtered_layers):
        raise ValueError('You are trying to load a weight file '
                         'containing ' + str(len(layer_names)) +
                         ' layers into a model with ' +
                         str(len(filtered_layers)) + ' layers.')"""

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        if "o0" in name:
            continue
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]
        layer = filtered_layers[k]
        symbolic_weights = layer.weights
        weight_values = preprocess_weights_for_loading(layer,
                                                       weight_values,
                                                       original_keras_version,
                                                       original_backend)
        if len(weight_values) != len(symbolic_weights):
            raise ValueError('Layer #' + str(k) +
                             ' (named "' + layer.name +
                             '" in the current model) was found to '
                             'correspond to layer ' + name +
                             ' in the save file. '
                             'However the new layer ' + layer.name +
                             ' expects ' + str(len(symbolic_weights)) +
                             ' weights, but the saved weights have ' +
                             str(len(weight_values)) +
                             ' elements.')
        weight_value_tuples += zip(symbolic_weights, weight_values)
    K.batch_set_value(weight_value_tuples)
