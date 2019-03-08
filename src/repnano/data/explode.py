import os
import h5py


def explode_fast5(name, tmpdir, n=None):
    # name : nom du fichier
    # tmpdir nom du repertoire temporaire
    # n: nombre de signaux extrait (None = tous)

    h5 = h5py.File(name, "r")
    os.makedirs(tmpdir, exist_ok=True)

    for ik, k in enumerate(h5.keys()):
        # print(k)
        h5p = h5[k]
        ch = int(str(h5p["channel_id"].attrs["channel_number"])[2:-1])
        readn = int(h5p["Raw"].attrs["read_number"])
        name = "read_%i_ch_%i_strand.fast5" % (readn, ch)
        # print(name)
        # break
        h5tmp = h5py.File(os.path.join(tmpdir, name), "w")
        h5tmp.create_group("Raw")
        raw = h5tmp["Raw"]
        # for el in ["Raw"]:#,"context_tags","tracking_id"]:
        for k in h5p["Raw"].attrs:
            # print("k",k,h5p[el].attrs[k])
            raw.attrs[k] = h5p["Raw"].attrs[k]
        #raw.attrs["read_id"] = h5p["Raw"].attrs["read_id"]
        read = raw.create_group("Reads")
        read2 = read.create_group("Read_%i" % readn)
        for k in h5p["Raw"].attrs:
            # print("k",k,h5p[el].attrs[k])
            read2.attrs[k] = h5p["Raw"].attrs[k]

        shape = h5p["Raw/Signal"].shape
        # print(np.median(h5p["Raw/Signal"].value),len(h5p["Raw/Signal"].value))
        read2.create_dataset("Signal", shape, dtype="<i2", data=h5p["Raw/Signal"].value)
        h5tmp.close()
        if n != None and ik >= n:
            break

    h5.close()


if __name__ == "__main__":
    import sys
    print(sys.argv)
    explode_fast5(sys.argv[1], tmpdir=sys.argv[2])


"""
root = "../../../deepnano5bases/data/tomb/"
fich = "env/ig/atelier/nanopore/cns/MN19358/RG_IN_PROCESS/190205_MN19358_FAK41381_A/fast5_file_management/workdir/barcode01/"
name = "/1ca9cbe0f2c9d4798f7e3ebf9c2ac6c7c775f0b4_0.fast5"

explode_fast5(root+fich+name,tmpdir="tmpd2")
"""
