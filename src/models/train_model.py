import sys
import numpy as np
import datetime
from collections import defaultdict
import os
#from sklearn.metrics import confusion_matrix
import glob
import keras
from Bio import pairwise2
import _pickle as cPickle


def print_stats(o):
    stats = defaultdict(int)
    for x in o:
        stats[x] += 1
    print(stats)


def flatten2(x):
    return x.reshape((x.shape[0] * x.shape[1], -1))


def find_closest(start, Index, factor=3.5):
    # Return the first element != N which correspond to the index of seqs
    start_index = min(int(start / factor), len(Index) - 1)
    # print(start,start_index,Index[start_index])
    if Index[start_index] >= start:
        while start_index >= 0 and Index[start_index] >= start:
            start_index -= 1
        return max(0, start_index)

    if Index[start_index] < start:
        while start_index <= len(Index) - 1 and Index[start_index] < start:
            start_index += 1
        if start_index <= len(Index) - 1 and start_index > 0:
            if abs(Index[start_index] - start) > abs(Index[start_index - 1] - start):
                start_index -= 1

            # print(start_index,Index[start_index])
        # print(start_index,min(start_index,len(Index)-1),Index[min(start_index,len(Index)-1)])
        return min(start_index, len(Index) - 1)


def get_segment(alignment, start_index_on_seqs, end_index_on_seqs):
    s1, s2 = alignment
    count = 0
    # print(s1,s2)
    startf = False
    end = None
    for N, (c1, c2) in enumerate(zip(s1, s2)):
        # print(count)
        if count == start_index_on_seqs and not startf:
            start = 0 + N
            startf = True
        if count == end_index_on_seqs + 1:
            end = 0 + N
            break

        if c2 != "-":
            count += 1

    # print(start,end)

    return s1[start:end].replace("-", ""), s1[start:end], s2[start:end]


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--Nbases', choices=["4", "5"], default='4')
    parser.add_argument('--root', type=str, default="data/training/")
    parser.add_argument('--test', dest='test', action='store_true')

    parser.add_argument('directories', type=str, nargs='*')
    args = parser.parse_args()

    data_x = []
    data_y = []
    data_y2 = []
    data_index = []
    data_alignment = []
    refs = []
    names = []

    if keras.backend.backend() != 'tensorflow':
        print("Must use tensorflow to train")
        exit()

    if args.Nbases == "4":
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}  # Modif
    elif args.Nbases == "5":
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "B": 4, "N": 5}  # Modif

    n_classes = len(mapping.keys())

    list_files = []
    subseq_size = 40

    for folder in args.directories:
        list_files += glob.glob(folder + "/*")

    list_files = list_files

    list_files.sort()

    os.makedirs(args.root, exist_ok=True)

    if not os.path.exists(os.path.join(args.root, "Allignements-bis")):
        end = None
        if args.test:
            end = 80

        for fn in list_files[:end]:
            print(fn)
            f = open(fn)
            ref = f.readline()
            ref = ref.replace("\n", "")
            if len(ref) > 30000:
                print("out", len(ref))
                continue

            X = []
            Y = []
            Y2 = []
            seq = []
            for l in f:
                its = l.strip().split()
                X.append(list(map(float, its[:-1])))
                Y.append(mapping[its[-1][0]])
                Y2.append(mapping[its[-1][1]])
                seq.append(its[-1])

            if len(X) < subseq_size:
                print("out (too small (to include must set a smaller subseq_size))", fn)
                continue
            refs.append(ref.strip())
            names.append(fn)
            data_x.append(np.array(X, dtype=np.float32))
            data_y.append(np.array(Y, dtype=np.int32))
            data_y2.append(np.array(Y2, dtype=np.int32))
            seq = "".join(seq)
            # print(seq)
            seq = seq[1::2]
            # print(seq)
            data_index.append(np.arange(len(seq))[np.array([s for s in seq]) != "N"])
            seqs = seq.replace("N", "")
            alignments = pairwise2.align.globalxx(ref, seqs)
            data_alignment.append(alignments[0][:2])
            #print(len(seqs), len(ref))
            print(len(alignments[0][0]), len(ref), len(seqs), alignments[0][2:])

        with open(os.path.join(args.root, "Allignements-bis"), "wb") as f:
            cPickle.dump([data_x, data_y, data_y2, data_index, data_alignment, refs, names], f)
    else:
        with open(os.path.join(args.root, "Allignements-bis"), "rb") as f:
            data_x, data_y, data_y2, data_index, data_alignment, refs, names = cPickle.load(f)

    print("done", sum(len(x) for x in refs))
    sys.stdout.flush()
    # print(len(refs[0]),len(data_x[0]),len(data_y[0]))
    # exit()

    s_arr = []
    p_arr = []
    for s in range(len(data_x)):
        s_arr += [s]
        p_arr += [len(data_x[s]) - subseq_size]

    sum_p = sum(p_arr)
    for i in range(len(p_arr)):
        p_arr[i] = 1. * p_arr[i] / sum_p

    batch_size = 1
    n_batches = len(data_x) / batch_size
    print(len(data_x), batch_size, n_batches, datetime.datetime.now())

    boring = False

    from .model import model2 as ntwk
    from .model import model as predictor


# ntwk.load_weights("./my_model_weights.h5")
    for epoch in range(10000):

        if epoch % 1000 == 0 and epoch != 0:
            if epoch != 0:
                predictor.load_weights(os.path.join(
                    args.root, '/my_model_weights-%i.h5' % (epoch - 1)))

            print("Realign")
            New_seq = []
            change = 0
            for s in range(len(data_x)):
                new_seq = np.argmax(predictor.predict(np.array([data_x[s]]))[0], axis=-1)
                # print(args.Nbases)
                if args.Nbases == "5":
                    alph = "ACGTTN"   # use T to Align
                if args.Nbases == "4":
                    alph = "ACGTN"
                New_seq.append("".join(list(map(lambda x: alph[x], new_seq))))
            # Here maybe realign with bwa
            # for s in range(len(data_x)):
                old_align = data_alignment[s]

                b = "B" in refs[s]
                ref = "" + refs[s]
                if b:
                    ref = ref.replace("B", "T")
                new_align = pairwise2.align.globalxx(ref, New_seq[s].replace("N", ""))[0][:2]
                print("Old", len(old_align[0]), "New", len(new_align[0]), b,)

                if len(new_align[0]) < len(old_align[0]):
                    print("Keep!")
                    change += 1
                    data_alignment[s] = new_align

                    data_index[s] = np.arange(len(New_seq[s]))[
                        np.array([ss for ss in New_seq[s]]) != "N"]
                else:
                    print()
            print("Change", change, len(data_x))
            import cPickle
            with open(os.path.join(
                    args.root, "Allignements-bis-%i" % epoch), "wb") as f:
                cPickle.dump([data_x, data_y, data_y2, data_index, data_alignment, refs, names], f)

                # Keep new alignment

        taken_gc = []
        out_gc = []
        tc = 0
        tc2 = 0
        tc3 = 0
        o1mm = []
        y1mm = []
        o2mm = []
        y2mm = []
        X_new = []
        Y_new = []
        Y2_new = []
        Label = []
        Length = []
        stats = defaultdict(int)
        for s in range(len(data_x)):
            s2 = np.random.choice(s_arr, p=p_arr)
            r = np.random.randint(0, data_x[s2].shape[0] - subseq_size)
            x = data_x[s2][r:r + subseq_size]
            # x[:,0] += np.random.binomial(n=1, p=0.1, size=x.shape[0]) *
            # np.random.normal(scale=0.01, size=x.shape[0])

            def domap(base):
                ret = [0 for b in range(n_classes)]
                ret[base] = 1
                return ret

            for xx in data_y2[s2][r:r + subseq_size]:
                stats[xx] += 1

            y = [domap(base) for base in data_y[s2][r:r + subseq_size]]
            y2 = [domap(base) for base in data_y2[s2][r:r + subseq_size]]

            if not boring:
                length = subseq_size
                start = r
                Index = data_index[s2]
                alignment = data_alignment[s2]

                start_index_on_seqs = find_closest(start, Index)
                end_index_on_seqs = find_closest(start + length, Index)
                #from IPython import embed
                # embed()
                #print(start, start_index_on_seqs, end_index_on_seqs, len(alignment))
                seg, ss1, ss2 = get_segment(alignment, start_index_on_seqs, end_index_on_seqs)

                maxi = 40
                l = min(max(len(seg), 1), maxi - 1)
                if abs(len(ss2.replace("-", "")) - len(ss2)) + abs(len(ss1.replace("-", "")) - len(ss1)) > 10:
                    continue
                Length.append(l)

                # print(len(s))
                if len(seg) > maxi - 1:
                    seg = seg[:maxi - 1]
                seg = seg + "A" * (maxi - len(seg))
                if "B" in refs[s2]:
                    seg = seg.replace("T", "B")
                # print(len(s))
                # print(s)
                # print([base for base in s])
                Label.append([mapping[base] for base in seg])
            X_new.append(x)
            Y_new.append(y)
            Y2_new.append(y2)

        X_new = np.array(X_new)
        Y_new = np.array(Y_new)
        Y2_new = np.array(Y2_new)
        Label = np.array(Label)
        Length = np.array(Length)
        print(X_new.shape, Y_new.shape)

        # To balance class weight

        print(Label)
        print(X_new.shape, Label.shape, np.array(
            [length] * len(Length)).shape, Length.shape)

        maxin = 10 * (int(len(X_new) // 10) - 3)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                      patience=5, min_lr=0.0001)
        # Log = keras.callbacks.CSVLogger(filename=os.path.join(
        #      args.root, "training.log"))

        print(len(data_x), np.mean(Length), np.max(Length))
        ntwk.fit([X_new[:maxin], Label[:maxin], np.array([subseq_size] * len(Length))[:maxin], Length[:maxin]],
                 Label[:maxin], nb_epoch=1, batch_size=10, callbacks=[reduce_lr],
                 validation_data=([X_new[maxin:maxin + 30],
                                   Label[maxin:maxin + 30],
                                   np.array([subseq_size] *
                                            len(Length))[maxin:maxin + 30],
                                   Length[maxin:maxin + 30]],
                                  Label[maxin:maxin + 30]))

        """
        import tensorflow as tf
        import keras.backend as K
        p = predictor.predict(X_new[:maxin])

        decoded, log_prob = K.ctc_decode(
            p, np.array([subseq_size] * len(Length))[:maxin])

        # Inaccuracy: label error rate

        ler = tf.reduce_mean(tf.edit_distance(
            tf.cast(decoded[0], tf.int32), K.ctc_label_dense_to_sparse(Label[:maxin], Length[:maxin])))
        print(ler)
        """

        # if epoch == 0:
        #    ntwk.fit(X_new,[Y_new,Y2_new],nb_epoch=1, batch_size=10,validation_split=0.05)
    if epoch % 10 == 0:
        ntwk.save_weights(os.path.join(
            args.root, '/my_model_weights-%i.h5' % epoch))
    """

    print epoch, tc / n_batches, 1. * tc2 / n_batches / batch_size, 1. * tc3 / n_batches / batch_size, datetime.datetime.now()
    print_stats(o1mm)
    print_stats(o2mm)
    print confusion_matrix(y1mm, o1mm)
    print confusion_matrix(y2mm, o2mm)"""

#  print "out", np.min(out_gc), np.median(out_gc), np.max(out_gc), len(out_gc)
