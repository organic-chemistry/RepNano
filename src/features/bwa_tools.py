
# read the ref file
import re


def LenghtOnRef(CIGAR):
    CIGARlenght = 0
    cigar = re.findall('\d+|\D+', CIGAR)
    for i in range(len(cigar) - 1):  # CorrL=L+delet-ins
        if cigar[i + 1] == 'S' or cigar[i + 1] == 'M' or cigar[i + 1] == 'D' or cigar[i + 1] == 'H':
            CIGARlenght += int(cigar[i])
        if cigar[i + 1] == 'I':
            CIGARlenght -= int(cigar[i])
    return CIGARlenght


def SeqInRef(Chrom, pos, bit, Lenght, ref):
    f = open(ref, 'r')
    s = f.readline()
    Ref = ''
    while s != '':

        if s.startswith('>'):
            if s[1:].split('\n')[0] == Chrom:
                s = f.readline()
                Ref = s.split()[0]
                s = f.readline()
                while not s.startswith('>'):
                    Ref += s.split()[0]
                    s = f.readline()
                    if s == '':
                        break
            else:
                s = f.readline()
        else:
            s = f.readline()
    if bit == '16' or bit == '2064':
        revcompl = lambda x: ''.join([{'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}[B] for B in x][::-1])
        #Ref = revcompl(Ref)
    f.close()
    SeqInRef = ''.join(Ref[pos:pos + Lenght])
    if bit == '16' or bit == '2064':
        return revcompl(SeqInRef)
    else:
        return SeqInRef


# read the sam file and write the input file for deepnano

def get_seq(sam, ref, ret_pos=False):
    ret = ["", 0, None, None]
    with open(sam, 'r') as f:
        s = f.readline()  # read the 1st line
        while s != '':
            ss = s.split()
            # print(ss)
            if ss[0] != '@SQ' and ss[0] != '@PG':
                if ss[1] == '0' or ss[1] == '16':  # take only main alignment (not chimeric)
                    Bit = ss[1]
                    Chrom = ss[2]
                    if ss[1] == '0':
                        pos = max(int(ss[3]) - 1, 0)
                    else:
                        pos = max(int(ss[3]), 0)

                    CIGAR = ss[5]
                    # SamSeq = ss[9]
                    # print ss[0]
                    # print Q
                    #print(Chrom, pos, Bit, LenghtOnRef(CIGAR), ref)
                    Seq = SeqInRef(Chrom, pos, Bit, LenghtOnRef(CIGAR), ref)
                    #print("Inside", Seq)
                    if ss[2] == '*' or "chr" not in ss[2]:
                        Chrom = None
                    else:
                        Chrom = int(ss[2][3:])
                    ret = [Seq, 1, Chrom, pos]

                else:

                    break

            s = f.readline()
    if ret_pos:
        return ret
    else:
        return ret[:2]


if __name__ == "__main__":
    SeqInRef("chr7", 557885, 0, 10088, "data/external/ref/S288C_reference_sequence_R64-2-1_20150113.fa")
