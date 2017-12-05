import sys
import numpy as np
import datetime
from collections import defaultdict
import os
# from sklearn.metrics import confusion_matrix
import glob
import keras
from Bio import pairwise2
import _pickle as cPickle
import copy
from ..features.helpers import scale_clean, scale_clean_two
from .helper import lrd
import csv
import keras.backend as K
from ..models.model_reverse import build_models


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--Nbases', type=int, choices=[4, 5, 8], default=4)
    parser.add_argument('--root', type=str, default="data/training/")
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--size', type=int, default=20)

    parser.add_argument('directories', type=str, nargs='*')

    parser.add_argument('--from-pre-trained', dest='from_pre_trained', action='store_true')
    parser.add_argument('--pre-trained-weight', dest='pre_trained_weight', type=str, default=None)
    parser.add_argument('--pre-trained-dir-list', dest='pre_trained_dir_list', type=str)
    parser.add_argument('--deltaseq', dest='deltaseq', type=int, default=10)
    parser.add_argument('--forcelength', dest='forcelength', type=float, default=0.5)
    parser.add_argument('--oversampleb', dest='oversampleb', type=int, default=3)
    parser.add_argument('--ctc', dest='ctc', action="store_true")
    parser.add_argument('--n-input', dest="n_input", type=int, default=1)
    parser.add_argument('--n-output', dest="n_output", type=int, default=1)
    parser.add_argument('--n-output-network', dest="n_output_network", type=int, default=1)
    parser.add_argument('--force-clean', dest="force_clean", action="store_true")
    parser.add_argument('--filter', nargs='+', dest="filter", type=str, default=[])
    parser.add_argument('--ctc-length', dest="ctc_length", type=int, default=20)
    parser.add_argument('--lr', dest="lr", type=float, default=0.01)
    parser.add_argument('--clean', dest="clean", action="store_true")
    parser.add_argument('--attention', dest="attention", action="store_true")
    parser.add_argument('--residual', dest="res", action="store_true")
    parser.add_argument('--all-file', nargs='+', dest="allignment_files", default=[], type=str)
    parser.add_argument('--simple', dest="simple", action="store_true")
    parser.add_argument('--all-T', dest="all_T", action="store_true")
    parser.add_argument('--hybrid', dest="hybrid", action="store_true")
    parser.add_argument('--feat', dest='feat', type=str)
    parser.add_argument('--hot', dest='hot', action="store_true")

    args = parser.parse_args()

    allf = args.allignment_files
    with open(allf[0], "rb") as f:
        seqs, signal = cPickle.load(f)

    with open(args.feat, "rb") as f:
        feat = cPickle.load(f)

    input_length = 100
    if not args.hot:
        model = build_models(input_length=input_length, n_feat=2)
    else:
        model = build_models(input_length=input_length, n_feat=5, hot=True)

    Schedul = lrd(waiting_time=100, start_lr=args.lr, min_lr=0.0001, factor=2)

    os.makedirs(args.root, exist_ok=True)

    for epoch in range(1000):
        X_new = []
        Y_new = []

        tr = 100
        for c in range(tr):
            choice = np.random.randint(len(seqs))
            if len(seqs[choice]) <= input_length:
                continue
            part = np.random.randint(len(seqs[choice]) - input_length)

            s_tmp = seqs[choice][part:part + input_length]
            if not args.hot:
                X_new.append([feat[st] for st in s_tmp])
            else:
                feat = {"A": [1, 0, 0, 0, 0], "C": [0, 1, 0, 0, 0],
                        "G": [0, 0, 1, 0, 0], "T": [0, 0, 0, 1, 0], "B": [0, 0, 0, 0, 1]}
                X_new.append([feat[st] for st in s_tmp])

            Y_new.append(signal[choice][part:part + input_length])

        X_new = np.array(X_new)
        Y_new = np.array(Y_new)

        r = model.fit(X_new, Y_new, nb_epoch=1, batch_size=10, validation_split=0.05)
        # ntwk.fit(X_new, Y_new, nb_epoch=1, batch_size=10, validation_split=0.05,
        #          sample_weight={"out_layer2": w2}, )

        csv_keys = ["epoch", "loss", "val_loss"]

        lr = Schedul.set_new_lr(r.history["loss"][0])
        K.set_value(model.optimizer.lr, lr)
        print(lr)

        if epoch == 0:
            with open(os.path.join(args.root, "training.log"), "w") as csv_file:
                writer = csv.writer(csv_file)
                # from IPython import embed
                # embed()
                # print(r)
                writer.writerow(csv_keys + ["lr"])
                writer.writerow([epoch] + [r.history[k][-1] for k in csv_keys[1:]] + [lr])
        else:
            with open(os.path.join(args.root, "training.log"), "a") as csv_file:
                writer = csv.writer(csv_file)
                # writer.writerow(k + ["lr"])
                writer.writerow([epoch] + [r.history[k][-1] for k in csv_keys[1:]] + [lr])
        if Schedul.stop:
            exit()

        # if epoch == 0:
        #    ntwk.fit(X_new,[Y_new,Y2_new],nb_epoch=1, batch_size=10,validation_split=0.05)
        if epoch % 10 == 0:
            model.save_weights(os.path.join(
                args.root, 'my_model_weights-%i.h5' % epoch))

#  print "out", np.min(out_gc), np.median(out_gc), np.max(out_gc), len(out_gc)
