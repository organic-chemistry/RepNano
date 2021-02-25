import mappy as mp
import sys

fasta = sys.argv[1]
index = sys.argv[2]


a = mp.Aligner(fasta, fn_idx_out=index)
