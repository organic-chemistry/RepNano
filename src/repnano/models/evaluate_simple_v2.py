from repnano.models.train_simple import load, create_model

if __name__ == "__main__":
    import argparse
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
    import os
    import numpy as np
    import logging, os

    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    from tensorflow.keras.models import load_model
    import tqdm
    from tensorflow.keras.utils import get_custom_objects

    logs = tf.keras.losses.LogCosh()


    @tf.function
    def loss1(y_true, y_pred, **kwargs):

        return logs(y_true, y_pred)


    def loss_mixture_b(y_true, y_pred):
        return y_pred

    get_custom_objects().update({'loss1': loss1,loss_mixture_b:'loss_mixture_b'})


    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)

    parser.add_argument('--model', type=str)
    parser.add_argument('--output', type=str,default="val")
    parser.add_argument('--error',action="store_true")
    parser.add_argument('--percent',action="store_true")
    parser.add_argument('--max_len',type=int,default=None)
    parser.add_argument('--plot',action="store_true")
    parser.add_argument('--final_size',type=int,default=100)
    parser.add_argument('--mods', nargs='+',type=str ,default=[""])
    parser.add_argument('--canonical', nargs='+',type=str ,default=[""])


    args = parser.parse_args()

    out_folder = os.path.split(args.output)[0]
    os.makedirs(out_folder,exist_ok=True)

    base_model = load_model(args.model)
    try:
        model = tf.keras.Model(inputs=base_model.input, outputs=[base_model.get_layer('percent').output,base_model.get_layer('B_percent').output])
        get_error = True

    except ValueError:
        try:
            model = tf.keras.Model(inputs=base_model.input, outputs=[base_model.get_layer('percent').output,
                                                                     base_model.get_layer('std_percent').output])
            get_error = True
        except:
            model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('percent').output)
            get_error = False

    model.summary()
    pad_size = (model.input[0].shape[-2] - args.final_size)//2

    data = load(args.file,per_read=True,pad_size=pad_size,final_size=args.final_size,max_read=args.max_len)
    h =[]
    std = []
    fasta = args.output

    print("Todo account for multiple")

    from collections import defaultdict
    equi = defaultdict(list)
    for i,(cano,sub) in enumerate(zip(args.canonical,args.mods)):
        equi[cano].append([i,sub])


    for X,y,read,f,sequence,extra in tqdm.tqdm(zip(data["X"][:args.max_len],data["y"],data["Readname"],data["Filename"],data["Sequences"],data["extra"])):
        #read ="/" + read
        r = model.predict(X)
        #print(r)
        if get_error:
            stda = r[1]
            stda = stda.reshape(-1,stda.shape[-1])
            std.append(np.mean(stda,axis=0))
            percent_std = np.zeros((len(sequence),stda.shape[-1]))
            percent_std[:len(stda)] = stda
            r = r[0]

        if not get_error:
            percent_std = None
        r = r.reshape(-1, r.shape[-1])
        #print(r.shape,len(sequence))
        percent = np.zeros((len(sequence),r.shape[1]))

        #r = r.flatten()
        percent[:len(r)] = r



        h.append(np.mean(r,axis=0))

        for k in equi.keys():
            proba = []
            subb = []
            for pos,base in equi[k]:
                proba.append(percent[:,pos])
                subb.append(base)
            proba = np.argmax(np.array(proba),axis=0)
            proba[np.max(proba,axis=0)<0.5]=-1
            for i,sub in enumerate(subb):
                 sequence[(sequence == k) & (proba == i)] = sub

        with open(fasta, "a") as fo:
            fo.writelines(">%s %s \n" % (read, str(extra)))
            fo.writelines("".join(sequence) + "\n")

        #print(h[-1])
        #print("Seq","".join([str(i) for i in sequence]))
        percent *= 100

        for i,base in enumerate(args.mods):
            #print(i,base)
            #print(percent)
            #print(percent_std)
            fastap = fasta + f"_ratio_{base}"
            fastap_std = fasta + f"_ratio_{base}_std"
            with open(fastap, "a") as fo1, open(fastap_std, "a") as fo_std:

                fo1.writelines(">%s\n" % read)
                fo1.writelines(" ".join(["%.2f" % ires2 for ires2 in percent[:,i]]) + "\n")
                if percent_std is not None:
                    fo_std.writelines(">%s\n" % read)
                    fo_std.writelines(" ".join(["%.2f" % ires2 for ires2 in percent_std[:,i]]) + "\n")

            if args.percent:
                onlyp = fasta + f"_percent_{base}"
                with open(onlyp, "a") as prc:
                    #print(i)
                    if percent_std is None:
                        percent_std = np.zeros((1,percent.shape[-1]))
                    prc.writelines(f"{read} {np.nanmean(percent[:,i]):.3f} {np.nanmean(percent_std[:,i]):.3f} {base}\n")

    import pylab
    import matplotlib as mpl

    mpl.use("Agg")
    h=np.array(h)
    std=np.array(std)
    #print(h.shape)
    #print(h[:10,])
    for i,base in enumerate(args.mods):
        nbin=100
        pylab.clf()
        pylab.hist(np.array(h[::,i]), range=[0, 1], bins=nbin)
        pylab.savefig(args.output[:-3]+f"distribution_{base}.png")
        if get_error:
            pylab.clf()
            pylab.figure()
            pylab.plot(h[::,i],np.array(std[::,i]),"o")
            pylab.plot([0,1],[0.045,0.045])
            pylab.xlabel("Percent")
            pylab.savefig(args.output[:-3]+f"distribution_std_{base}.png")
            pylab.figure()

            pylab.hist(np.array(h[::,i])[np.array(std[::,i])<0.045], range=[0, 1], bins=nbin)
            pylab.savefig(args.output[:-3] + f"distribution_filtered_{base}.png")




