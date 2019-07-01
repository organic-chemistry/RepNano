# LSTM and CNN for sequence classification in the IMDB dataset

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, AveragePooling1D, TimeDistributed, Dropout, Input, Bidirectional
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
import _pickle as cPickle
import numpy as np
import os


def create_model(params,
                 rootw="/data/bioinfo@borvo/users/jarbona/mongo_net/first/",
                 create_only=False, wn="", typem=1):
    # typem=1, kernel_size=3, filters=32, neurones=100,
    #             activation="linear", pooling=True, mpool=2, dropout=0):
    init = 5
    if typem == 1:

        model = Sequential()
        model.add(Conv1D(filters=int(params['filters']),
                         kernel_size=int(params['kernel_size']),
                         padding='same',
                         activation='relu', input_shape=(160, init)))
        if params["choice_pooling"]['pooling']:
            model.add(MaxPooling1D(pool_size=params["choice_pooling"]["pool_size"]))
        if params['dropout'] != 0:
            model.add(Dropout(params['dropout']))
        if params["bi"]:
            model.add(Bidirectional(LSTM(int(params['neurones']))))
        else:
            model.add(LSTM(int(params['neurones'])))

        if params["nc"] == 1:
            model.add(Dense(1, activation=params['activation']))
            if params.get("cost","logcosh") == "custom":
                print("Custom loss")
                def normal(x,p,sigma):
                    return 1/(2*sigma)**0.5*K.exp(-(x-p)**2/(2*sigma)**2)
                def loss(y_true,y_pred):
                    sigma=0.2
                    return K.mean(1/(2*sigma)**0.5 - normal(y_true,0,sigma) - normal(y_true,y_pred,sigma))

                model.compile(loss=loss, optimizer=params['optimizer'])
            else:


                model.compile(loss=params.get("cost","logcosh"), optimizer=params['optimizer'])
        else:
            input = Input(shape=(160, init))
            LSTMo = model(input)
            B = Dense(1, activation=params['activation'])(LSTMo)
            E = Dense(1, activation=params['activation'])(LSTMo)

            global_model = Model(inputs=input, outputs=[B, E])
            global_model.compile(loss='mse', optimizer=params['optimizer'])
            model = global_model

    elif typem == 3:
        seq = Input(shape=(160, 5))
        inside = Bidirectional(LSTM(40, return_sequences=True))(seq)
        Brdu = TimeDistributed(Dense(1, activation="sigmoid"))(inside)
        Idu = TimeDistributed(Dense(1, activation="sigmoid"))(inside)
        model = Model(inputs=seq, outputs=[Brdu, Idu])
        funcType = type(model.predict)

        def predict(self, X):
            r1, r2 = Model.predict(self, X)
            m = np.ones_like(r1) * X[::, ::, 2:3]  # Keep T
            m[m == 0] = np.nan
            # print(r1[0,:40])
            r1[r1 > 0.5] = 1
            r1[r1 < 0.5] = 0
            r2[r2 > 0.5] = 1
            r2[r2 < 0.5] = 0
            # print(r1[0,:40])
            # print(m[0,:40])
            r1 = r1 * m
            r2 = r2 * m
            return np.nanmean(r1, axis=1), np.nanmean(r2, axis=1)
        model.predict = funcType(predict, model)

        # model.load_weights("test_longueur_lstm_from_scratch_without_human/weights.25-0.02.hdf5")
        # model.load_weights("test_longueur/weights.05-0.02.hdf5")
    else:
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                         activation='relu', input_shape=(160, init)))
        """
        model.add(MaxPooling1D(pool_size=4)) # 16
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=4)) #4
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                                 activation='relu'))

        # model.add(LSTM(100))
        # model.add(Dense(1, activation='linear'))
        """
        model.add(MaxPooling1D(pool_size=4))
        model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        model.add(AveragePooling1D(pool_size=10))
        model.add(Flatten())
        model.compile(loss='logcosh', optimizer='adam')
    # model.load_weights("test_cnv2/weights.18-0.03.hdf5")

    if create_only:
        return model

    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

    def fl(name):
        if type(name) == dict:
            return "".join(["%s-%s" % (p, str(fl(value))) for p, value in name.items()])
        else:
            return name
    name = fl(params)
    print(name)

    if wn != "":
        print("Loading", wn)
        model.load_weights(wn)

    checkpointer = ModelCheckpoint(
        filepath=rootw+'weights_%s.hdf5' % name,
        verbose=1, save_best_only=True)
    es = EarlyStopping(patience=10)
    print("Loading from", rootw)
    X_train = np.load(os.path.join(rootw, "X_train.npy"))
    y_train = np.load(os.path.join(rootw, "y_train.npy"))
    X_val = np.load(os.path.join(rootw, "X_val.npy"))
    y_val = np.load(os.path.join(rootw, "y_val.npy"))

    print(np.mean(y_train, axis=0))

    if params["nc"] == 1:
        model.fit(X_train, y_train[::, 0], epochs=40,
                  batch_size=int(params['batch_size']),
                  sample_weight=y_train[::, 1],
                  validation_split=0.1, callbacks=[checkpointer, es])
        # Final evaluation of the model

        scores = model.evaluate(X_val, y_val[::, 0], verbose=0)
        print(scores)
        return {'loss': -scores, 'status': STATUS_OK}
    else:
        """
        y_train[::, 1][y_train[::, 1] == 0] = 1
        y_train[::, 3][y_train[::, 3] == 0] = 1
        print(np.mean(y_train, axis=0))"""

        model.fit(X_train, [y_train[::, 0], y_train[::, 2]], epochs=40,
                  batch_size=int(params['batch_size']),
                  sample_weight=[y_train[::, 1], y_train[::, 3]],
                  validation_split=0.1, callbacks=[checkpointer, es])
        # Final evaluation of the model

        scores = model.evaluate(X_val, y_val[::, 0], verbose=0)
        print(scores)
