import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

import pandas as pd
import h5py
import numpy as np
import pylab
import matplotlib as mpl
mpl.use("Agg")
try:
    from tensorflow_probability import distributions as tfd
except:
    pass
import functools
"""
inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

"""


import scipy.special

def define_mixture(N):
    bn = tf.constant(np.array([scipy.special.binom(N,k) for k in range(N+1)]),dtype=tf.float32)
    def binomiale(p,N,bn):
        return tf.math.pow(p,tf.range(1.0*N+1)) * tf.math.pow(1-p,tf.range(1.0*N,-1,-1)) * bn
    #plot(binomiale(0.1,50,bn=bn))
    def bp(p):
        return binomiale(p,N,bn)

    def mixture(alphas,probs):
        #alphas_b = tf.broadcast_to(alphas,
        #                                (1,tf.shape(alphas)[0]))
        alphasn = alphas / tf.reduce_sum(alphas)
        probs=tf.cast(probs,dtype=tf.float32)
        return tf.reduce_sum(tf.vectorized_map(bp,probs) * alphasn[:,tf.newaxis] ,axis=0)
    return mixture



def loss_mixture(y_true, y_pred,batch_size,mixture,plot=None):
    # Kernel density treatment for y_pred

    s = tf.shape(y_true)
    d0= tf.gather(s,0)
    d1= tf.gather(s,1)


    up = tf.transpose(tf.reshape(tf.meshgrid(tf.range(d0), tf.range(d1//2)),(2,-1)),name="up")
    down = tf.transpose(tf.reshape(tf.meshgrid(tf.range(d0), tf.range(d1//2,d1)),(2,-1)),name="down")
    down1 = tf.transpose(tf.reshape(tf.meshgrid(tf.range(d0), tf.range(d1//2+1,d1)),(2,-1)),name="down1")


    alphas = tf.gather_nd(y_true, up)
    p = tf.gather_nd(y_true, down)

    real_p = tf.cast(tf.gather_nd(y_true, down1),dtype=tf.float32)

    y_p_c = tf.keras.backend.clip(tf.reshape(y_pred,[-1]), 1e-7, 1-1e-7)

    gm_pred = mixture(alphas=y_p_c, probs=y_p_c)
    gm_true = mixture(alphas=alphas, probs=p)



    #v = tf.keras.losses.KLDivergence()(gm_true,gm_pred)
    gm_true = tf.keras.backend.clip(gm_true, 1e-7, 1)
    gm_pred = tf.keras.backend.clip(gm_pred, 1e-7, 1)
    loss_kld = tf.reduce_mean(gm_true * tf.math.log(gm_true / gm_pred), axis=-1)

    if plot is not None:
        tf.print("y_true", y_true.shape)
        tf.print("y_pred", y_pred.shape)

        tf.print(gm_pred.shape)
        tf.print(gm_true.shape)
        pylab.plot(gm_pred, label=f"pred {plot[:-4]} + loss {loss_kld}")
        pylab.plot(gm_true, label="True " + plot[:-4])
        pylab.legend()
        pylab.savefig(plot)

    return loss_kld #tf.keras.losses.LogCosh()(real_p,y_pred) #+ tf.reduce_mean(gm_true * tf.math.log(gm_true / gm_pred), axis=-1)
    #tf.print("y_true2")
    #tf.print(y_true2,len(y_true2))
    #tf.print(y_pred2,len(y_pred2))

    #v = tf.reduce_sum(y_true2 * tf.math.log(y_true2 / y_pred2))
    #tf.print(v,v.shape)
    #return 0.1 *  v / batch_size + tf.keras.losses.log_cosh(real_p,y_pred)
    return v / batch_size #+ tf.keras.losses.LogCosh()(real_p,y_pred)
def create_model(error=True,lstm=False,final_size=100,informative=False,nmod=1,mixture=False,batch_size=32):

    kernel_size = 7

    n_layers = 3
    n_channel = 5
    target_output_size = final_size
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


    if informative:
        input_info = tf.keras.Input(shape=(final_size,n_channel))

    output_mod = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(nmod,activation="sigmoid"),name="percent")(tmp)

    if informative:
        output_mod *= input_info

    fit_output_mod = tf.keras.layers.AveragePooling1D(pool_size=final_size)(output_mod)


    if not error:
        inputs = [input]
        if informative:
            inputs += [input_info]
        model = tf.keras.Model(inputs=inputs,outputs=fit_output_mod)
        if mixture:
            #opt = tf.keras.optimizers.Adam(clipvalue=0.1,learning_rate=0.0001)

            def loss_mixture_b(y_true,y_pred):
                return loss_mixture(y_true,y_pred,batch_size=batch_size,mixture=define_mixture(50))
            model.compile(optimizer="Adam", loss=loss_mixture_b)
        else:
            model.compile(optimizer="Adam", loss="LogCosh")

    else:
        input_mod_pred = tf.keras.Input(shape=(nmod))

        output_delta_mod = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(nmod,activation="sigmoid"),name="std_percent")(tmp)
        fit_output_delta_mod = tf.keras.layers.AveragePooling1D(pool_size=final_size)(output_delta_mod)
        fit_output_delta_mod = (fit_output_mod-input_mod_pred)**2 - fit_output_delta_mod
        if informative:
            fit_output_delta_mod *= input_info

        logs = tf.keras.losses.LogCosh()
        @tf.function
        def loss1(y_true,y_pred,**kwargs):

            return logs(y_true,y_pred)
            #loss1 = logs()



        inputs = [input,input_mod_pred]
        if informative:
            inputs += [input_info]

        model = tf.keras.Model(inputs=inputs,outputs=[fit_output_mod,fit_output_delta_mod])
        model.compile(optimizer="Adam", loss=[logs,loss1])

    model.summary()
    return model



from repnano.features.extract_events import get_events
from repnano.models.simple_utilities import transform_read
import pandas as pd



def get_type(h5):
    if "Reads" in h5.keys():
        return "mega"
    return "rep"

def smooth(ser,sc):
    return np.array(pd.Series(ser).rolling(sc, min_periods=1, center=True).mean())

def window_stack_numpy_v2(a,stepsize,width):
    stride = a.strides[-1]
    last_dim=a.shape[-1]
    nline = int((len(a)-width)/(stepsize) + 1)

    return np.lib.stride_tricks.as_strided(a, shape=(nline,width*last_dim), strides=(stepsize*last_dim*stride,stride))

def weighted_smooth(a, sm, weight=None):
    if len(a.shape) == 1:
        a = a.reshape(-1, 1)

    add = sm
    extra_start = np.cumsum(a[:add - 1]) / np.arange(1, add)
    extra_end = np.cumsum(a[::-1][:add - 1]) / np.arange(1, add)

    if add % 2 == 1:
        rm = 0
    else:
        rm = 1
    extra_start = extra_start[add // 2 - rm:]
    extra_end = extra_end[add // 2:][::-1]

    if weight is not None:
        if len(weight.shape) == 1:
            weight = weight.reshape(-1, 1)
        wgw = window_stack_numpy_v2(weight, stepsize=1, width=sm)
        # print(wgw.shape)
        wgw /= np.sum(wgw, axis=1)[:, None]
        ws = wgw * window_stack_numpy_v2(a, stepsize=1, width=sm)
        ws = np.sum(ws, axis=1)


    else:
        ws = window_stack_numpy_v2(a, stepsize=1, width=sm).mean(axis=1)
    return np.concatenate([extra_start, ws, extra_end])

def standardize_name(read_name):
    if "read" in read_name:
        return read_name.split("_")[1]
    elif read_name.startswith("/"):
        read_name=read_name[1:]
    return read_name

def iterate_over_h5(h5,typef,stride=5):
    if typef=="rep":
        for read_name in h5.keys():
            #print(read_name)
            events, rawV, sl = get_events(h5[read_name],tomb=True,bigf=True)
            yield standardize_name(read_name) , events, rawV, sl
    else:
        alphabet=h5.attrs["alphabet"]
        print(alphabet)
        for k in h5["Reads"]:
            read = h5["Reads"][k]

            exp = dict(read.attrs.items())
            # print(exp)
            exp["Reference"] = np.array(read["Reference"])
            exp["Ref_to_signal"] = np.array(read["Ref_to_signal"])[:-1] # it is one longer than bases
            assert(len(exp["Ref_to_signal"]) == len(exp["Reference"]))
            exp["Dacs"] = np.array(read["Dacs"])
            current_pA = (exp["Dacs"] + exp["offset"]) * exp["range"] / exp["digitisation"]
            current = (current_pA - exp["shift_frompA"]) / exp["scale_frompA"]
            maxi =len(current)-1
            stop = exp["Ref_to_signal"] > maxi
            if np.sum(stop) == 0:
                maxi= None
            else:
                maxi = np.argmax(stop)-1
            classical = True
            if classical:

                current = weighted_smooth(current,5)

                bases = np.zeros(len(exp["Reference"]),dtype=str)
                for i,l in enumerate(alphabet):
                    bases[exp["Reference"]==i]=l

                yield standardize_name(exp["read_id"]) ,{"mean":current[exp["Ref_to_signal"][:maxi]],"bases":bases[:maxi]},"",{}
            else:
                ref_to_s = exp["Ref_to_signal"][:maxi]
                maxt =  ref_to_s[-1]

                current = current[:maxt+1]
                bases = np.zeros(len(current),dtype=str)
                bases[:] = "N"
                for i, l in enumerate(alphabet):
                    bases[ref_to_s[exp["Reference"]==i]]=l
                assert(len(current)==len(bases))
                yield standardize_name(exp["read_id"]), {"mean": current,
                                                         "bases": bases}, "", {}







def iter_keys(h5,typef):
    if typef=="rep":
        return h5.keys()
    else:
        return [h5["Reads"][k].attrs["read_id"] for k in h5["Reads"] ]


def load_bigf_with_percents(data_frame,name_bigf,max_read=None,mods=[]):
    X = []
    y = []

    h5 = h5py.File(name_bigf, "r")
    typef=get_type(h5)
    #print(name_bigf)
    #for read_name,percent in zip(data_frame.readname,data_frame.percent):

    for read_name, events, rawV, sl in iterate_over_h5(h5, typef=typef):
        #print(read_name)
        found = data_frame["readname"] == read_name
        if np.sum(found) == 0:
            continue
        else:
            for index, row in data_frame[found].iterrows():
                break
        read_name = row["readname"]
        percent = np.array([row[f"percent_{m}"] for m in mods])
        #print(row)
        try:
            error =np.array([row[f"error{m}"] for m in mods])
        except:
            error=np.array([0]*len(mods))
        #print(events,read_name)
        X.append({"mean":events["mean"], "bases": events["bases"],"readname":read_name,"filename":name_bigf,"extra":sl,"error":error})
        y.append(percent)
        if max_read is not None and len(y) == max_read:
            break

    return X,y
def load_bigf(name_bigf,max_read=None):
    X = []
    y = []
    sl = []
    h5 = h5py.File(name_bigf, "r")

    typef=get_type(h5)

    for read_name , events, rawV, sl in iterate_over_h5(h5,typef=typef):

        X.append({"mean":events["mean"], "bases": events["bases"],"readname":read_name,"filename":name_bigf,"extra":sl})
        y.append(np.array([np.nan]))
        if max_read is not None and len(y) == max_read:
            break

    return X,y

def load(file,per_read=False,pad_size=12,max_read=None,final_size=100,mixture=False,mods=[]):
    window_size=final_size + 2 * pad_size

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
    stride = []
    for name_bigf in nf:
        if p is not None:
            X,y = load_bigf_with_percents(data_frame=p[p.file_name==name_bigf],
                            name_bigf=name_bigf,max_read=max_read,mods=mods)
        else:
            X,y = load_bigf(name_bigf=name_bigf,max_read=max_read)
        #print(len(X),len(y))
        for ir,(xv,yv) in enumerate(zip(X,y)):
            if len(xv["mean"])< window_size:
                continue
            if max_read is not None and ir > max_read:
                print("Reach limit",max_read)
                break
            x_t,y_t = transform_read(xv,yv,window_size=window_size,pad_size=pad_size)
            y_t /= 100
            DataX.append(x_t)
            if mixture:
                y_tmp = np.zeros((y_t.shape[0],4)) # two first component are mixture values the the two next are probability values
                if np.sum(y_t) == 0:
                    y_tmp[:,:2] = 0.5
                    #y_tmp[:]
                else:
                    y_tmp[:,0]=0.2
                    y_tmp[:,1]=0.8
                    y_t /= 0.8
                    y_t[y_t>1] = 1
                    y_tmp[:,3] = y_t

                y_t = y_tmp
            Datay.append(y_t)
            Readname.append(xv["readname"])
            Filename.append(xv["filename"])
            Sequences.append(xv["bases"])
            Extra.append(xv["extra"])
            #print(xv)
            if "error" in xv.keys():
                Error.append(xv["error"]*np.ones_like(y_t))
            else:
                Error.append(np.zeros_like(y_t))
    #print(Error)
    if per_read:
        return {"X":DataX,"y":Datay,"Readname":Readname,"Filename":Filename,"Sequences":Sequences,"extra":Extra}
    else:
        #print(len(DataX),len(Datay),len(Error))
        return {"X": np.concatenate(DataX, axis=0), "y": np.concatenate(Datay, axis=0),"error":np.concatenate(Error,axis=0),"extra":Extra}


def load_data(list_files,pad_size,max_read=None,final_size=100,mixture=False,mods=[]):
    X = []
    y = []
    error = []
    extra = []
    for file in list_files:
        print("Loading",file)
        intermediary = load(file,pad_size=pad_size,max_read=max_read,final_size=final_size,mixture=mixture,mods=mods)
        X.append(intermediary["X"])
        y.append(intermediary["y"])
        error.append(intermediary["error"])
        extra.append(intermediary["extra"])

    return {"X":np.concatenate(X,axis=0),"y":np.concatenate(y,axis=0),"error":np.concatenate(error,axis=0),"extra":extra}



def unison_shuffled_copies(a, b,error=None):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    if error is None:
        return a[p], b[p]
    else:
        return a[p], b[p],error[p]

def load_percent(list_percent,pad_size=12,thres_error=0.045,max_read=None,final_size=100,mixture=False,mods=[]):

    data  = load_data(list_percent,pad_size=pad_size,max_read=max_read,final_size=final_size,mixture=mixture,mods=mods)
    X=data["X"]
    y=data["y"]
    error = data["error"]
    extra = data["extra"]

    X,y,error = unison_shuffled_copies(X,y,error)
    y = y[:,np.newaxis,:]


    error[error>thres_error] = 0.1
    error[error<=thres_error] = 1

    #print(X.shape,y.shape,error.shape)
    nans = np.any(np.any(np.isnan(X),axis=-1),axis=-1) | np.any(np.isnan(y),axis=-1)[:,0] | np.any(np.isnan(error),axis=-1)
    X=X[~nans]
    y=y[~nans]
    error=error[~nans]
    extra=[ e for e,good in zip(extra,~nans) if good]

    return X,y,error,extra


if __name__ == "__main__":
    import argparse
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
    import os
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data', type=str)
    parser.add_argument('--root_save', type=str)
    parser.add_argument('--weights', type=str)

    parser.add_argument('--percents_training', nargs='+' ,type=str,default=["Brdu_0.00"])
    parser.add_argument('--percents_validation', nargs='+',type=str,default=[])
    parser.add_argument('--final_size',type=int,default=100)
    parser.add_argument('--stride',type=int,default=5)
    parser.add_argument('--batch_size',type=int,default=32)


    parser.add_argument('--max_len',type=int,default=None)

    parser.add_argument('--validation',type=float,default=0.1)
    parser.add_argument('--error',action="store_true")
    parser.add_argument('--smalllr',action="store_true")
    parser.add_argument('--lstm',action="store_true")
    parser.add_argument('--weight',action="store_true")
    parser.add_argument('--multi_gpu',action="store_true")
    parser.add_argument('--mixture',action="store_true")
    parser.add_argument('--weigths',type=str)
    parser.add_argument('--mods', nargs='+',type=str ,default=[""])
    parser.add_argument('--epochs', type=int,default=100)




    args = parser.parse_args()

    if args.multi_gpu:
        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():

            model = create_model(error=args.error,lstm=args.lstm,final_size=args.final_size)
    else:
        model = create_model(error=args.error, lstm=args.lstm,
                             final_size=args.final_size,mixture=args.mixture,
                             batch_size=args.batch_size,nmod=len(args.mods))

    if args.weights:
        def loss_mixture_b(y_true,y_pred):
            return y_pred


        from tensorflow.keras.utils import get_custom_objects

        get_custom_objects().update({'loss_mixture_b': loss_mixture_b})

        model_load = tf.keras.models.load_model(args.weights)
        #weights = model_load.get_weights()
        model_load.save_weights("/tmp/weights.h5")
        model.load_weights("/tmp/weights.h5",skip_mismatch=True,by_name=True)



    root_data=args.root_data
    root_save = args.root_save +"/"
    pad_size = (model.input[0].shape[-2] - args.final_size)//2
    print(pad_size)

    X,y,sw,extra= load_percent(list_percent=[root_data+f"/{p}/percent.csv" for p in args.percents_training],
                          pad_size=pad_size,max_read=args.max_len,final_size=args.final_size,mixture=args.mixture,mods=args.mods)
    print("Mean not excluded",np.mean(sw))
    #exit()

    if args.percents_validation != []:
        Xv,yv,swv,extra = load_percent( list_percent=[root_data+f"/{p}/percent.csv" for p in args.percents_validation],
                                  max_read=args.max_len,
                                  pad_size=pad_size,final_size=args.final_size,mixture=args.mixture,mods=args.mods)
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

        target = [y,np.zeros_like(y)]
        X = [X,y]

        target_val = (yv,np.zeros_like(yv))
        Xv = (Xv,yv)

    else:
        target = y
        target_val = yv

    if args.smalllr:
        callbacks = [checkpointer,
                     EarlyStopping(patience=10),
                     CSVLogger(root_save + 'log.csv'),
                     ReduceLROnPlateau(patience=5)]
    else:
        callbacks = [checkpointer,
                     EarlyStopping(patience=3),
                     CSVLogger(root_save + 'log.csv')]

    def truncv(v):
        return v
    if args.mixture:
        def truncv(v):
            return v[:int(args.batch_size * (len(v)//args.batch_size))]


    #check for nan:
    #print(type(X))
    if np.sum(np.isnan(y.flatten())) != 0 :
        raise "Nan in y"
    if not args.error:
        if np.sum(np.isnan(X.flatten())) != 0:
            raise "Nan in X"
    else:
        if np.sum(np.isnan(X[0].flatten())) != 0 or np.sum(np.isnan(X[1].flatten())) != 0 :
            raise "Nan in X"



    if args.weight:
        model.fit(X,target,
                  validation_data = (Xv,target_val,swv),
                  sample_weight = sw,
                  epochs=100,
                  callbacks=callbacks)
    else:
        if args.mixture:
            print(target[:100,3])
            print(model.predict(X[:args.batch_size]))
            loss_mixture(np.array(target[:args.batch_size],dtype=np.float32), model.predict(X[:args.batch_size]),
                         batch_size=args.batch_size,mixture=define_mixture(50),plot="start.png")

        else:
            pass
            #print(len(target),target[:10],target.shape)

        model.fit(truncv(X),truncv(target),
                  validation_data = (truncv(Xv),truncv(target_val)),
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  callbacks=callbacks)
    if args.mixture:
        p = model.predict(X[:args.batch_size])
        for pred,ptrue in zip(p,target):
            print(pred,ptrue)

        loss_mixture(np.array(target[:args.batch_size],dtype=np.float32), model.predict(X[:args.batch_size]),
                             batch_size=args.batch_size,mixture=define_mixture(50),plot="end.png")

    #model.save("first_model")


