import h5py
import glob
from multiprocessing import Pool
import argparse
import numpy as np

parser.add_argument('--root', type=str, default="data/training/")
parser.add_argument("--n-cpu", dest="n_cpu", type=int, default=None)
parser.add_argument("--ws", dest="ws", type=int, default=5)
args = parser.parse_args()



def add_segment(filename,ws):

    wsi = np.random.randint(ws,ws+3)
    h5 = h5py.File(filename, "a")
    try:
        events = get_events(h5, already_detected=False,
                            chemistry="rf", window_size=ws, old=False,verbose=False,about_max_len=about_max_len)
    except:
        return 1

    Segmentation = h5.create_group('/Segmentation_Rep')
    Segmentation.create_dataset(name='events', data=events)
    Segmentation.create_dataset(name="window_size",data=ws,dtype="int")
    h5.close()

    return 0


    with Pool(n_cpu) as p:
        res = p.map(load_from_bc, D.strands)


files = glob.glob(args.root)

def Add_segment(filename):
    return add_segment(filename,ws=args.ws)


with Pool(n_cpu) as p:
    res = p.map(load_from_bc, files)

print(np.mean(res))
