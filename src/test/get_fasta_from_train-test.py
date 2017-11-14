import sys
out_network = sys.argv[1]
get_from = sys.argv[2]
out_name = sys.argv[3]

Files = []

if get_from != "all":
    with open(get_from, "r") as f:
        for line in f.readlines():
            Files.append(line.split()[0])

cut = []
cutT = []
B = 0
L = 0
E = 0
I = 0
T = 0
U = 0
found = []
lost = []
with open(out_network, "r") as f:
    data = f.readlines()
    for fi, seq in zip(data[::2], data[1:][::2]):

        doit = False
        for Fi in Files:
            if Fi in fi:
                doit = True
                found.append(Fi)
                break
        if get_from == "all":
            doit = True
        if doit:

            cut.append(fi + seq)

            U += seq.count("U")
            B += seq.count("B")
            L += seq.count("L")
            E += seq.count("E")
            I += seq.count("I")

            T += seq.count("T")
            seq = seq.replace("U", "T")
            seq = seq.replace("B", "T")
            seq = seq.replace("L", "T")
            seq = seq.replace("E", "T")
            seq = seq.replace("I", "T")

            cutT.append(fi + seq)


with open(out_name + ".fasta", "w") as f:
    f.writelines("".join(cut))

with open(out_name + "_T" + ".fasta", "w") as f:
    f.writelines("".join(cutT))

print(len(Files), len(cut), len(cutT))
print(out_name)
r = 0
if T != 0:
    r = B / 1.0 / T

print("Number of T %i, number of B %i, ratio %f" % (T, B, r))
r = 0
if T != 0:
    r = L / 1.0 / T

print("Number of T %i, number of L %i, ratio %f" % (T, L, r))
r = 0
if T != 0:
    r = E / 1.0 / T

print("Number of T %i, number of E %i, ratio %f" % (T, E, r))
r = 0
if T != 0:
    r = I / 1.0 / T

print("Number of T %i, number of I %i, ratio %f" % (T, I, r))

r = 0
if T != 0:
    r = U / 1.0 / T

print("Number of T %i, number of U %i, ratio %f" % (T, U, r))
