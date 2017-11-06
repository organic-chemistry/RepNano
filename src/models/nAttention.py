import tensorflow as tf
# Code from https://github.com/datalogue/keras-attention/edit/master/models/custom_recurrents.py
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent, _time_distributed_dense
from keras.engine import InputSpec

tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)


class AttentionDecoder(Recurrent):

    def __init__(self, units, output_dim,
                 activation='tanh',
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Implements an AttentionDecoder that takes in a sequence encoded by an
        encoder and outputs the decoded states
        :param units: dimension of the hidden state and the attention matrices
        :param output_dim: the number of labels in the output space

        references:
            Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
            "Neural machine translation by jointly learning to align and translate."
            arXiv preprint arXiv:1409.0473 (2014).
        """
        self.units = units
        self.output_dim = output_dim
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.window_length = 5

        super(AttentionDecoder, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # must return sequences

    def build(self, input_shape):
        """
          See Appendix 2 of Bahdanau 2014, arXiv:1409.0473
          for model details that correspond to the matrices here.
        """

        self.batch_size, self.timesteps, self.input_dim = input_shape

        if self.stateful:
            super(AttentionDecoder, self).reset_states()

        self.states = [0]  # y, s

        """
            Matrices for creating the context vector
        """

        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        """
            Matrices for making the final prediction vector
        """
        self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim),
                                   name='C_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_o = self.add_weight(shape=(self.output_dim, ),
                                   name='b_o',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        self.input_spec = [
            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        self.built = True

    def call(self, x):
        # store the whole sequence so we can "attend" to it at each timestep
        self.x_seq = x

        # apply the a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # thefore we can save computation time:
        self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                             input_dim=self.input_dim,
                                             timesteps=self.timesteps,
                                             output_dim=self.units)

        # if self.window_length is not None:
    #        self._uxpb = K.temporal_padding(self._uxpb, (self.window_length, self.window_length))
        if K.backend() == 'tensorflow':

            #self._uxpb = tfPrint("_uxpb", self._uxpb)

            # equivalent to K.expand_dims(self.x_seq) but it does not work for training
            #xr = K.reshape(self.x_seq, (-1, 1, self.timesteps, self.input_dim))
            xr = K.expand_dims(self.x_seq, 1)

            self._window_uxpb = tf.extract_image_patches(xr,
                                                         ksizes=(
                                                             1, 1, 2 * self.window_length + 1, 1),
                                                         strides=(1, 1, 1, 1), rates=(1, 1, 1, 1),
                                                         padding="SAME")
            self._window_uxpb = K.squeeze(self._window_uxpb, 1)
            #self._window_uxpb = tfPrint("_window_uxpb", self._window_uxpb)

        return super(AttentionDecoder, self).call(self._window_uxpb)

    def get_initial_state(self, inputs):

        return [0.]

    def step(self, x, states):

        pos = states[0]
        # this relates how much other timesteps contributed to this one.
        wl = 2 * self.window_length + 1
        x = K.reshape(x, (-1, wl, self.input_dim))
        #x = tfPrint("_x", x)
        xpb = _time_distributed_dense(x, self.U_a, b=self.b_a,
                                      input_dim=self.input_dim,
                                      timesteps=wl,
                                      output_dim=self.units)

        # Exponential window:
        """
        exp_win = K.exp(- (K.arange(0, K.shape(self._uxpb)
                                    [1], 1, dtype="float32") - pos)**2 / self.window_length)

        exp_win = K.expand_dims(exp_win, 0)
        exp_win = K.expand_dims(exp_win, -1)
        """

        # exp_win = K.ones_like(self._uxpb)
        et = K.dot(activations.tanh(xpb),
                   K.expand_dims(self.V_a))
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, wl)
        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, x, axes=1), axis=1)

        yt = activations.softmax(K.dot(context, self.C_o) + self.b_o)

        if self.return_probabilities:
            return at, [pos + 1.]
        else:
            return yt, [pos + 1.]

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# check to see if it compiles
if __name__ == '__main__':
    from keras.layers import Input, LSTM
    from keras.models import Model
    from keras.layers.wrappers import Bidirectional
    i = Input(shape=(100, 104), dtype='float32')
    enc = Bidirectional(LSTM(64, return_sequences=True), merge_mode='concat')(i)
    dec = AttentionDecoder(32, 4)(enc)
    model = Model(inputs=i, outputs=dec)
    model.summary()
