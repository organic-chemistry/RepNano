
import glob
from ..features.bwa_tools import get_seq
from ..features.extract_events import get_raw, extract_events
import h5py
import numpy as np
from Bio import pairwise2
import pylab


class NotAllign(Exception):

    def __init__(self, delta):
        self.message = "Delta = % i" % delta

REF = "data/external/ref/S288C_reference_sequence_R64-2-1_20150113.fa"


class Dataset:

    def __init__(self, samfile, root_files):
        self.root_files = root_files
        self.samfile = samfile

    def populate(self, maxf=None, minion=True, basecall=True, filter_not_alligned=False, filter_ch=None):
        self.strands = []
        lstrand = glob.glob(self.root_files + "/*")

        def find_strand(strand):
            for st in lstrand:
                if strand in st:
                    return st

        with open(self.samfile, "r") as f:
            for line in f.readlines():
                if maxf is not None and len(self.strands) >= maxf:
                    break
                sp = line.split()
                if len(sp) > 1 and sp[0].startswith("@ch"):
                    # print(sp[:4])
                    if "chr" in sp[2]:
                        X = int(sp[2][3:])
                    else:
                        X = 0

                    fn = "read_%s_ch_%s_" % (sp[0].split("_")[1][4:], sp[0].split("_")[0][3:])
                    # print(fn)
                    # print(find_strand(fn))

                    if filter_ch is not None and X not in filter_ch:
                        print("Skip in ch %i " % X)
                        continue

                    self.strands.append(Strand(find_strand(fn),
                                               X_from_Minion=int(X),
                                               t_from_Minion=0,
                                               sam_line_minion=line))
                    if minion:
                        self.strands[-1].get_seq(f="Minion")
                    if basecall:
                        try:
                            self.strands[-1].get_seq(f="BaseCall")
                        except NotAllign:
                            print("Strand %i raw not alligned to basecall" % len(self.strands))

                            if filter_not_alligned:
                                print("Removed")
                                self.strands.pop(-1)

            """
            chp = kp.split("_")[0][3:]
            readp =  kp.split("_")[1][4:]
"""

"""
def compare_metrichor(file_netw,type_b="B",show_identical=False,ntwk=None):

    if type_b == "T":
        contrl = "BTF_AG_ONT_1_FAH14273_A-select.sam"
    if type_b == "B":
        contrl = "BTF_AH_ONT_1_FAH14319_A-select.sam"
    if type_b == "L":
        contrl = "BTF_AI_ONT_1_FAH14242_A-select.sam"
    if type_b == "E":
        contrl = "BTF_AK_ONT_1_FAH14211_A-select.sam"
    if type_b == "I":
        contrl = "BTF_AL_ONT_1_FAH14352_A-select.sam"

    with open(file_netw +"/"+ contrl.replace("select","select_pass_test_T"),"r") as f:
        datan = {}
        for line in f.readlines():
            sp = line.split()

            if len(sp) > 1 and sp[0].startswith("data"):
                datan[sp[0].split("/")[-1]] = sp[1:4]
            else:
                pass
                #print(sp)

"""


class Strand:

    def __init__(self, filename, X_from_Minion=0, t_from_Minion=0, sam_line_minion=""):
        self.filename = filename
        self.X_from_Minion = X_from_Minion
        self.t_from_Minion = t_from_Minion
        self.sam_line_minion = sam_line_minion

    def get_seq(self, f):
        if f == "Minion":
            seq, sucess = get_seq(self.sam_line_minion, ref=REF, from_line=True)
            self.seq_from_minion = seq

        if f == "BaseCall":
            h5 = h5py.File(self.filename, "r")
            e2bis = h5["Analyses/Basecall_1D_000/BaseCalled_template/Events"]

            s2 = ""
            self.signal_bc = []
            left = None
            for ie, (s, length, m, ms, move, stdv) in enumerate(zip(e2bis["start"], e2bis["length"],
                                                                    e2bis["mean"], e2bis["model_state"], e2bis["move"], e2bis["stdv"])):

                state = "%s" % ms.tostring()
                state = state[2:-1]  # to remove b' and '
                # print(s)
                #state = state[2]

                if move == 1:
                    s2 += state[2]
                    state = state[2]
                    self.signal_bc.append((state, m, stdv, length, s))
                    left = None
                elif move >= 2:
                    s2 += state[1:3]
                    state = state[1:3]

                    if self.signal_bc[-1][0] == "N":
                        self.signal_bc[-1] = list(self.signal_bc[-1])
                        self.signal_bc[-1][0] = state[0]
                        self.signal_bc[-1] = tuple(self.signal_bc[-1])

                        self.signal_bc.append((state[1], m, stdv, length, s))
                    else:
                        self.signal_bc.append((state[0], m, stdv, length, s))
                        left = state[1]
                elif move == 0:
                    self.signal_bc.append(("N", m, stdv, length, s))
                    if left is not None:
                        self.signal_bc[-1] = list(self.signal_bc[-1])
                        self.signal_bc[-1][0] = left
                        self.signal_bc[-1] = tuple(self.signal_bc[-1])
                    left = None
                else:
                    left = None

            self.signal_bc = np.array(self.signal_bc, dtype=[('seq', np.str_, 2), ('mean', 'f4'),
                                                             ('stdv', 'f4'), ('length', 'f4'), ('start', 'f4')])

            self.allign_basecall_raw()

            self.signal_bc["start"] += (self.imin / self.sampling_rate - self.signal_bc["start"][0])

            self.seq_from_basecall = s2

    def score_ref(self, maxlen=1000):
        s1 = self.seq_from_minion
        s2 = self.seq_from_basecall
        self.score(s1, s2, maxlen=maxlen)

    def score(self, s1, s2, maxlen=1000):

        if s1 == "":
            return None
        if s2 == "":
            return None

        if maxlen is not None:
            s1 = s1[:maxlen]
            s2 = s2[:maxlen]

        al = pairwise2.align.globalxs(s1, s2, -0.5, -0.5, one_alignment_only=True)[0][2]

        return al / len(s1)

    def allign_basecall_raw(self):

        h5 = h5py.File(self.filename, "r")
        raw, amp = get_raw(h5)

        e2bis = h5["Analyses/Basecall_1D_000/BaseCalled_template/Events"]
        to_match = np.zeros(int((e2bis["length"][-1] + e2bis["start"]
                                 [-1] - e2bis["start"][0]) * amp)) * np.nan

        s0 = e2bis["start"][0]

        for ie, (s, l, m) in enumerate(zip(e2bis["start"], e2bis["length"], e2bis["mean"])):
            to_match[int((s - s0) * amp):int(((s + l - s0) * amp))] = m

        mini = 100000
        lmatch = len(to_match)
        imin = 0
        for i in range(len(raw) - lmatch)[:5000]:
            delta = np.nanmean((raw[i:i + lmatch] - to_match)**2)
            if delta < mini:
                imin = i + 0
                mini = delta + 0
                # print(delta)

        if mini > 100:
            raise NotAllign(mini)
        # print(delta,imin)
        self.imin = imin
        self.to_match = to_match
        self.raw = raw
        self.sampling_rate = amp

    def show_segmentation_bc(self):
        self.allign_basecall_raw()
        l = len(self.to_match)
        x = np.arange(l) + self.imin

        pylab.plot(self.raw[:l + self.imin])

        pylab.plot(*self.segmentation_to_plot(self.signal_bc, shift=self.imin))

    def segmentation_to_plot(self, sign, shift=None, sl=6024):
        X = []
        Y = []
        s0 = 0
        if shift is not None:
            s0 = sign["start"][0]
        else:
            shift = 0
        for i, (s, l, m) in enumerate(zip(sign["start"], sign["length"], sign["mean"])):
            X += [(s - s0) * sl + shift, (s - s0 + l) * sl + shift]
            Y += [m, m]

        return np.array(X), np.array(Y)

    def plot_sequence(self, sign, window=[None, None], color="k", up=5):
        sl = self.sampling_rate
        for i, (s, l, m, base) in enumerate(zip(sign["start"], sign["length"], sign["mean"], sign["seq"])):

            if (window[0] is None or (window[0] is not None and s * sl > window[0])) and \
                    (window[1] is None or (window[1] is not None and s * sl < window[1])):
                pylab.text((s + l / 2) * sl - 1.5, m + up, base,
                           color=color)  # 1.5 size of character

    def segmentation(self, chem="rf", w=5):
        h5 = h5py.File(self.filename, "r")
        self.segments = extract_events(h5, chem="rf", window_size=w)

    def transfer(self, root_signal, signal_to_label, center_of_mass=False):
        # Compute center:

        r_center = root_signal["start"] + root_signal["length"] / 2
        stl_center = signal_to_label["start"] + signal_to_label["length"] / 2

        stl_base = []

        if center_of_mass:

            for c in stl_center:
                where = np.argmin(np.abs(r_center - c))
                stl_base.append(root_signal["seq"][where])

        else:
            # Force assignation of bases
            stl_base = ["N" for i in range(len(signal_to_label))]
            for ib, b in enumerate(root_signal["seq"]):
                if b != "N":

                    s = (r_center[ib] > signal_to_label["start"]) &  \
                        (r_center[ib] < (signal_to_label["start"] + signal_to_label["length"]))

                    #where = np.argmin( np.abs(stl_center-r_center[ib]))
                    where = np.argmax(s)
                    stl_base[where] = b

        new_signal = []
        for s, m, std, l, start in zip(stl_base, signal_to_label["mean"], signal_to_label["stdv"],
                                       signal_to_label["length"], signal_to_label["start"]):
            new_signal.append((s, m, std, l, start))

        return np.array(new_signal, dtype=[('seq', np.str_, 2), ('mean', 'f4'),
                                           ('stdv', 'f4'), ('length', 'f4'), ('start', 'f4')])

"""
ref = "../../data/processed/results/ctc_20170908-R9.5/BTF_AG_ONT_1_FAH14273_A-select_pass_test_T.sam"
new = "../../data/processed/results/v9p5-new_ctc_20170908-R9.5/BTF_AG_ONT_1_FAH14273_A-select_pass_test_T.sam"
root = "../../data/raw/20170908-R9.5/"
D = Dataset(samfile = root + "BTF_AG_ONT_1_FAH14273_A-select.sam",
            root_files=root+"AG-basecalled/")
D.populate(maxf=10,filter_not_alligned=True,filter_ch =range(1,11) )
s = D.strands[0]"""
