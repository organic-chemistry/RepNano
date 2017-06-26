"""
Script to split the dataset into training and
testing.
Here we choose the chromosome 11 to be the test
need bwa
"""


import os
import subprocess


def split(filename, ref, output, exclude=[11]):

    seqs = []
    with open(filename, "r") as f:
        for line in f.readlines():
            name = line.split()[0]
            sequence = line.split()[1]
            seqs.append(">%s\n%s" % (name, sequence))
    # print(seqs)
    tmpfile = "/tmp/seq.fa"
    with open(tmpfile, "w") as f:
        f.writelines("\n".join(seqs))
    #ref = os.path.join(os.getcwd(), ref)
    print(ref)

    subprocess.call("bwa mem -x ont2d > /tmp/aligned %s  %s " % (ref, tmpfile), shell=True)

    Ch = []
    with open("/tmp/aligned", "r") as ch:
        for line in ch.readlines():
            if line.startswith("@"):
                pass
            else:
                try:
                    Ch.append(int(line.split()[2][3:]))
                except:
                    print(line.split()[2])
                    Ch.append("Unkwon")
    Train = []
    Test = []
    with open(filename, "r") as init:
        for line, ch in zip(init.readlines(), Ch):
            # print(line)
            if ch in exclude:
                Test.append(line)
            else:
                Train.append(line)

    with open(output + ".train", "w") as f:
        f.writelines("".join(Train))
    with open(output + ".test", "w") as f:
        f.writelines("".join(Test))

    print("Proportion test", len(Test) / 1.0 / len(Train))


if __name__ == "__main__":
    import sys
    split(filename=sys.argv[1], ref=sys.argv[2], exclude=[11])
