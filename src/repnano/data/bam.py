import pysam
import numpy as np

def convert_to_coordinate(seq, Ml, Mm, which="T"):
    # Ml base proba
    # Mm number of base to skip
    Mmc = Mm.copy()
    result = np.zeros((len(seq))) + np.nan
    n_which = 0
    for bi, s in enumerate(seq):
        if s == which:
            # print(bi,len(seq))
            if n_which > len(Mmc) - 1:
                break
            skip = Mmc[n_which]
            if skip == 0:
                result[bi] = Ml[n_which] / 255
                n_which += 1
            else:
                Mmc[n_which] -= 1
    return result


def load_read_bam(bam, filter_b=0.5, n_b=500):
    samfile = pysam.AlignmentFile(bam, "r")  # ,check_sq=False)

    Read = {}

    for ir, read in enumerate(samfile):
        # print(ir)

        seq, Ml, Mm = read.get_forward_sequence(), read.get_tag("Ml"), [int(v) for v in
                                                                        read.get_tag("Mm")[:-1].split(",")[1:]]
        attr = {}
        if read.is_reverse:
            attr["mapped_strand"] = "-"
        else:
            attr["mapped_strand"] = "+"
        attr["mapped_chrom"] = "chr%i" % read.reference_id
        pos = read.get_reference_positions()
        attr["mapped_start"] = pos[0]
        attr["mapped_end"] = pos[-1]
        attr["seq"] = seq
        Nn = convert_to_coordinate(seq, Ml, Mm)
        if np.sum(np.array(Nn) > filter_b) > n_b:
            Read["read_" + read.query_name] = [attr, Nn]
    return Read