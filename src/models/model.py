from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import backend as K
from keras.layers.core import Lambda
from keras.optimizers import SGD
import keras

if keras.backend.backend() == 'tensorflow':
    import tensorflow as tf

    def ctc_batch_cost(y_true, y_pred, input_length, label_length, ctc_merge_repeated=False):
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


inputs = Input(shape=(None, 4))
Nbases = 5 + 1
size = 20

l1 = Bidirectional(LSTM(size, return_sequences=True), merge_mode='concat')(inputs)
l2 = Bidirectional(LSTM(size, return_sequences=True), merge_mode='concat')(l1)
l3 = Bidirectional(LSTM(size, return_sequences=True), merge_mode='concat')(l2)
out_layer1 = TimeDistributed(Dense(Nbases, activation="softmax"), name="out_layer1")(l3)

try:
    model = Model(input=inputs, output=out_layer1)
except:
    model = Model(inputs=inputs, outputs=out_layer1)

model.compile(optimizer='adadelta', loss='categorical_crossentropy', sample_weight_mode='temporal')

if keras.backend.backend() == 'tensorflow':
    old = True
    if not old:
        def ctc_lambda_func(y_true, y_pred):
            #y_pred, labels
            y_pred = y_pred[:, :, :]
            return ctc_batch_cost(labels, y_pred, input_length, label_length, ctc_merge_repeated=False)

        labels = Input(name='the_labels', shape=[40], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        """
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
            [out_layer1, labels, input_length, label_length])"""

        model2 = Model(inputs=inputs, outputs=out_layer1)

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        #rms = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0, clipvalue=0.05)
        model2.compile(loss={'out_layer1': ctc_lambda_func}, optimizer=sgd)

    if old:
        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            y_pred = y_pred[:, :, :]
            return ctc_batch_cost(labels, y_pred, input_length, label_length, ctc_merge_repeated=False)

        labels = Input(name='the_labels', shape=[40], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
            [out_layer1, labels, input_length, label_length])

        model2 = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        #rms = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0, clipvalue=0.05)
        model2.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
