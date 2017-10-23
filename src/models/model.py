from keras.layers import Input, Dense
from keras.layers.merge import Concatenate

from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import backend as K
from keras.layers.core import Lambda, Reshape
from keras.optimizers import SGD, Adadelta
import keras


def build_models(size=20, nbase=1, trainable=True, ctc_length=40, ctc=True,
                 uniform=True, input_length=None, n_output=1, n_feat=4, recurrent_dropout=0):
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

    l1 = Bidirectional(LSTM(size, return_sequences=True, trainable=trainable, recurrent_dropout=recurrent_dropout),
                       merge_mode='concat')(inputs)
    l2 = Bidirectional(LSTM(size, return_sequences=True, trainable=trainable, recurrent_dropout=recurrent_dropout),
                       merge_mode='concat')(l1)

    if n_output == 1:
        l3 = Bidirectional(LSTM(size, return_sequences=True, trainable=trainable, recurrent_dropout=recurrent_dropout),
                           merge_mode='concat')(l2)

        out_layer1 = TimeDistributed(Dense(Nbases, activation="softmax"), name="out_layer1")(l3)

    else:
        l3 = Bidirectional(LSTM(2 * size, return_sequences=True, trainable=trainable),
                           merge_mode='concat')(l2)

        if input_length != None:
            TD = TimeDistributed(Dense(Nbases, activation="softmax"), name="out_layer1")
            r = Reshape((input_length * 2, 2 * size))(l3)
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

    if out_layer2 is not None:
        model = Model(inputs=inputs, outputs=[out_layer1, out_layer2])
    else:
        model = Model(inputs=inputs, outputs=out_layer1)

    ada = Adadelta(lr=.2, rho=0.95, epsilon=1e-08, decay=0.0)
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

            model2 = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

            sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
            # rms = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0, clipvalue=0.05)
            model2.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    return model, model2
