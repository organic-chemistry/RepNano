from keras.layers import Input, Dense
from keras.layers.merge import Add

from keras.models import Model
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.optimizers import Adadelta
from keras.layers.core import Reshape

try:
    from .attention import AttentionDecoder
except:
    print("Loading theano no attention")
# from .attentionBis import Attention


def build_models(size=4, nbase=1, trainable=True, ctc_length=40, ctc=True,
                 uniform=True, input_length=None, n_output=1,
                 n_feat=4, recurrent_dropout=0, lr=0.01, res=False, attention=False, simple=False, w=10, hot=False):

    input_length = input_length

    if not hot:
        inputs = Input(shape=(input_length, w, w, n_feat))

        print("Trainable ???", trainable)

        merge_mode = "sum"

        from keras.layers import ConvLSTM2D as LSTM

        l1 = Bidirectional(LSTM(size,
                                kernel_size=3, padding="same", return_sequences=True, data_format="channels_last",
                                trainable=trainable, recurrent_dropout=recurrent_dropout),
                           merge_mode=merge_mode)(inputs)
        # if res:
        # l1 = Add()([l1, inputs])
        l2 = Bidirectional(LSTM(size,
                                kernel_size=3, padding="same", return_sequences=True, data_format="channels_last",
                                trainable=trainable, recurrent_dropout=recurrent_dropout),
                           merge_mode=merge_mode)(l1)
        l2 = Add()([l2, l1])
        l2 = Reshape((input_length, w * w * size))(l2)

        out_layer1 = TimeDistributed(Dense(1, activation="linear"), name="out_layer1")(l2)

        model = Model(inputs=inputs, outputs=out_layer1)

        ada = Adadelta(lr=lr, rho=0.95, epsilon=1e-08, decay=0.0)

        model.compile(optimizer='adadelta', loss='MSE')

        return model

    else:

        inputs = Input(shape=(input_length, n_feat))

        print("Trainable ???", trainable)

        merge_mode = "sum"

        from keras.layers.recurrent import LSTM

        l1 = Bidirectional(LSTM(size, return_sequences=True,
                                trainable=trainable, recurrent_dropout=recurrent_dropout),
                           merge_mode=merge_mode)(inputs)
        # if res:
        # l1 = Add()([l1, inputs])
        l2 = Bidirectional(LSTM(size, return_sequences=True,
                                trainable=trainable, recurrent_dropout=recurrent_dropout),
                           merge_mode=merge_mode)(l1)
        l2 = Add()([l2, l1])
        #l2 = Reshape((input_length, w * w * size))(l2)

        out_layer1 = TimeDistributed(Dense(2, activation="linear"), name="out_layer1")(l2)

        model = Model(inputs=inputs, outputs=out_layer1)

        ada = Adadelta(lr=lr, rho=0.95, epsilon=1e-08, decay=0.0)

        model.compile(optimizer='adadelta', loss='MSE')

        return model
