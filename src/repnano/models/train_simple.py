import tensorflow as tf
import pandas as pd
import h5py
import numpy as np

"""
inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

"""
def create_model(error=True,lstm=False):

    kernel_size = 7

    n_layers = 4
    n_channel = 5
    target_output_size = 100
    filter_size = 32 #32
    change_filter_size = 1.6  # 2 for three layers
    if not lstm:
        input_shape = target_output_size + n_layers * (kernel_size-1)
    else:
        input_shape = target_output_size + (n_layers-1) * (kernel_size-1)

    input = tf.keras.Input(shape=(input_shape,n_channel))
    tmp = input
    for i in range(n_layers):
        if i != n_layers - 1 or ((i==n_layers -1) and not lstm):
            tmp  = tf.keras.layers.Conv1D(filters=int(filter_size * change_filter_size **(i+1)),
                                          kernel_size=kernel_size, padding='valid',
                                         activation='relu')(tmp)
        else:
            tmp = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=int(filter_size * change_filter_size ** (i + 1)/2),
                                       return_sequences=True),merge_mode="concat")(tmp)


    output_B = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1,activation="sigmoid"),name="B_percent")(tmp)
    fit_output_B = tf.keras.layers.Flatten()(tf.keras.layers.AveragePooling1D(pool_size=100)(output_B))

    if not error:
        model = tf.keras.Model(inputs=input,outputs=fit_output_B)
        model.compile(optimizer="Adam", loss="logcosh")

    else:
        input_b_pred = tf.keras.Input(shape=(1))

        output_delta_B = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1,activation="sigmoid"),name="delta_B_percent")(tmp)
        fit_output_delta_B = tf.keras.layers.Flatten()(tf.keras.layers.AveragePooling1D(pool_size=100)(output_delta_B))
        fit_output_delta_B = (fit_output_B-input_b_pred)**2 - fit_output_delta_B

        logs = tf.keras.losses.LogCosh()
        @tf.function
        def loss1(y_true,y_pred,**kwargs):

            return logs(y_true,y_pred)
            #loss1 = logs()

        model = tf.keras.Model(inputs=[input,input_b_pred],outputs=[fit_output_B,fit_output_delta_B])
    model.compile(optimizer="Adam", loss=[logs,loss1])

    model.summary()
    return model



from repnano.features.extract_events import get_events
from repnano.models.simple_utilities import transform_read

def load_bigf_with_percents(data_frame,name_bigf):
    X = []
    y = []
    h5 = h5py.File(name_bigf, "r")
    #print(name_bigf)
    #for read_name,percent in zip(data_frame.readname,data_frame.percent):
    for index, row in data_frame.iterrows():
        read_name = row["readname"]
        percent = row["percent"]
        #print(row)
        try:
            error = row["error"]
        except:
            error=0
        v = h5[read_name]
        #print(v)
        #print(error)
        events, rawV, sl = get_events(v,tomb=True,bigf=True)

        X.append({"mean":events["mean"], "bases": events["bases"],"readname":read_name,"filename":name_bigf,"extra":sl,"error":error})
        y.append(percent)

    return X,y
def load_bigf(name_bigf):
    X = []
    y = []
    h5 = h5py.File(name_bigf, "r")
    for read_name in h5.keys():
        v = h5[read_name]

        events, rawV, sl = get_events(v,tomb=True,bigf=True)

        X.append({"mean":events["mean"], "bases": events["bases"],"readname":read_name,"filename":name_bigf,"extra":sl})
        y.append(np.nan)

    return X,y

def load(file,per_read=False,pad_size=12):
    window_size=100 + 2 * pad_size

    DataX = []
    Datay = []
    Readname = []
    Filename = []
    Sequences = []
    Extra = []
    Error = []
    if file.endswith("csv"):
        p = pd.read_csv(file)
        nf = set(p["file_name"])
    else:
        p=None
        nf = [file]

    for name_bigf in nf:
        if p is not None:
            X,y = load_bigf_with_percents(data_frame=p[p.file_name==name_bigf],
                            name_bigf=name_bigf)
        else:
            X,y = load_bigf(name_bigf=name_bigf)
        for xv,yv in zip(X,y):
            if len(xv["mean"])< window_size:
                continue
            x_t,y_t = transform_read(xv,yv,window_size=window_size,pad_size=pad_size)
            DataX.append(x_t)
            Datay.append(y_t)
            Readname.append(xv["readname"])
            Filename.append(xv["filename"])
            Sequences.append(xv["bases"])
            Extra.append(xv["extra"])
            #print(xv)
            if "error" in xv.keys():
                Error.append(xv["error"]*np.ones_like(y_t))
    #print(Error)
    if per_read:
        return {"X":DataX,"y":Datay,"Readname":Readname,"Filename":Filename,"Sequences":Sequences,"extra":Extra}
    else:
        return {"X": np.concatenate(DataX, axis=0), "y": np.concatenate(Datay, axis=0),"error":np.concatenate(Error,axis=0)}


def load_data(list_files,pad_size):
    X = []
    y = []
    error = []
    for file in list_files:
        print("Loading",file)
        intermediary = load(file,pad_size=pad_size)
        X.append(intermediary["X"])
        y.append(intermediary["y"])
        error.append(intermediary["error"])

    return {"X":np.concatenate(X,axis=0),"y":np.concatenate(y,axis=0),"error":np.concatenate(error,axis=0)}



def unison_shuffled_copies(a, b,error=None):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    if error is None:
        return a[p], b[p]
    else:
        return a[p], b[p],error[p]

def load_percent(list_files_percent,pad_size=12,thres_error=0.045):
    list_percent = [root_data+f"/percent_{p}.csv" for p in list_files_percent]
    data  = load_data(list_percent,pad_size=pad_size)
    X=data["X"]
    y=data["y"]
    error = data["error"]

    X,y,error = unison_shuffled_copies(X,y,error)


    error[error>thres_error] = 0.1
    error[error<=thres_error] = 1


    return X,y,error


if __name__ == "__main__":
    import argparse
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
    import os
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data', type=str)
    parser.add_argument('--root_save', type=str)
    parser.add_argument('--percents_training', nargs='+' ,type=int,default=[0,20,40,60,80,100])
    parser.add_argument('--percents_validation', nargs='+' ,type=int,default=[])

    parser.add_argument('--validation',type=float,default=0.1)
    parser.add_argument('--error',action="store_true")
    parser.add_argument('--smalllr',action="store_true")
    parser.add_argument('--lstm',action="store_true")
    parser.add_argument('--weight',action="store_true")



    args = parser.parse_args()

    model = create_model(error=args.error,lstm=args.lstm)

    root_data=args.root_data
    root_save = args.root_save +"/"
    pad_size = (model.input[0].shape[-2] - 100)//2
    print(pad_size)

    X,y,sw = load_percent(args.percents_training,pad_size=pad_size)
    print("Mean excluded",np.mean(sw))


    if args.percents_validation != []:
        Xv,yv,swv = load_percent(args.percents_validation,pad_size=pad_size)
        nv = min(int(len(X) * args.validation), len(Xv))
        print(nv)
        Xv = Xv[:nv]
        yv = yv[:nv]
        swv=swv[:nv]

    else:

        nv = int(len(X) * args.validation)
        print(nv)
        Xv = X[-nv:]
        yv = y[-nv:]
        swv=sw[-nv:]
        X = X[:-nv]
        y = y[:-nv]
        sw=sw[:-nv]


    print(Xv.shape,yv.shape,X.shape,y.shape)

    p,n = os.path.split(args.root_save)
    os.makedirs(p,exist_ok=True)

    checkpointer = ModelCheckpoint(
        filepath=root_save + 'weights.hdf5' ,
        verbose=1, save_best_only=True)

    if args.error:

        target = [y/100,np.zeros_like(y)]
        X = [X,y/100]

        target_val = (yv/100,np.zeros_like(yv))
        Xv = (Xv,yv/100)

    else:
        target = y / 100
        target_val = yv / 100

    if args.smalllr:
        callbacks = [checkpointer,
                     EarlyStopping(patience=10),
                     CSVLogger(root_save + 'log.csv'),
                     ReduceLROnPlateau(patience=5)]
    else:
        callbacks = [checkpointer,
                     EarlyStopping(patience=5),
                     CSVLogger(root_save + 'log.csv')]

    if args.weight:
        model.fit(X,target,
                  validation_data = (Xv,target_val,swv),
                  sample_weight = sw,
                  epochs=100,
                  callbacks=callbacks)
    else:
        model.fit(X,target,
                  validation_data = (Xv,target_val),
                  epochs=100,
                  callbacks=callbacks)
    #model.save("first_model")


