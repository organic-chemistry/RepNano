from repnano.models.train_simple import load, create_model

if __name__ == "__main__":
    import argparse
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
    import os
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import tqdm
    from tensorflow.keras.utils import get_custom_objects

    logs = tf.keras.losses.LogCosh()


    @tf.function
    def loss1(y_true, y_pred, **kwargs):

        return logs(y_true, y_pred)

    get_custom_objects().update({'loss1': loss1})


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






    args = parser.parse_args()

    out_folder = os.path.split(args.output)[0]
    os.makedirs(out_folder,exist_ok=True)

    base_model = load_model(args.model)
    model = tf.keras.Model(inputs=base_model.input, outputs=[base_model.get_layer('B_percent').output,base_model.get_layer('delta_B_percent').output])
    model.summary()
    pad_size = (model.input[0].shape[-2] - args.final_size)//2

    data = load(args.file,per_read=True,pad_size=pad_size,final_size=args.final_size,max_read=args.max_len)
    h =[]
    std = []
    fasta = args.output
    fastap = fasta+"_ratio_B"
    fastap_std = fasta + "_ratio_B_std"
    onlyp = fasta+"_percentBrdu"
    with open(fasta,"w") as fo, open(fastap,"w") as fo1, open(fastap_std,"w") as fo_std, open(onlyp,"w") as prc:

        for X,y,read,f,sequence,extra in tqdm.tqdm(zip(data["X"][:args.max_len],data["y"],data["Readname"],data["Filename"],data["Sequences"],data["extra"])):
            #read ="/" + read
            r = model.predict(X)
            if len(r) == 2:
                std.append(np.mean(r[1]))
                stda = r[1].flatten()
                Brdu_std = np.zeros(len(sequence))
                Brdu_std[:len(stda)] = stda
                r = r[0]
            Brdu = np.zeros(len(sequence))

            r = r.flatten()
            Brdu[:len(r)] = r

            h.append(np.mean(r))
            #print(h[-1])
            #print("Seq","".join([str(i) for i in sequence]))
            sequence[(sequence=="T") & (Brdu >0.5)] = "B"

            fo.writelines(">%s %s \n" % (read, str(extra)))
            fo.writelines("".join(sequence) + "\n")

            fo1.writelines(">%s\n" % read)
            fo1.writelines(" ".join(["%.2f" % ires2 for ires2 in Brdu]) + "\n")

            fo_std.writelines(">%s\n" % read)
            fo_std.writelines(" ".join(["%.2f" % ires2 for ires2 in Brdu_std]) + "\n")

            if args.percent and std != []:
                prc.writelines(f"{read} {np.nanmean(Brdu):.3f} {np.nanmean(Brdu_std):.3f}\n")

    import pylab
    import matplotlib as mpl

    mpl.use("Agg")

    nbin=100
    pylab.hist(np.array(h), range=[0, 1], bins=nbin)
    pylab.savefig(args.output[:-3]+"distribution.png")
    if std != []:
        pylab.figure()
        pylab.plot(h,np.array(std),"o")
        pylab.plot([0,1],[0.045,0.045])
        pylab.xlabel("Percent")
        pylab.savefig(args.output[:-3]+"distribution_std.png")
        pylab.figure()

        pylab.hist(np.array(h)[np.array(std)<0.045], range=[0, 1], bins=nbin)
        pylab.savefig(args.output[:-3] + "distributio_filtered.png")




