import h5py
import glob


def concat(name, tmpdir, n=None):
    error_al = 0
    error_template = 0
    with h5py.File(name, "r+") as h5:
        ls = glob.glob(tmpdir+"/*")
        for f in ls:
            with h5py.File(f, "r") as tmp:
                idi = tmp["Raw"].attrs["read_id"].decode()
                # print(idi)
                read = h5["read_"+idi]
                try:
                    read = read.create_group("BaseCalled_template")
                except ValueError:
                    read = h5["read_"+idi+"/BaseCalled_template"]
                    pass

                try:
                    for k in tmp["Analyses/RawGenomeCorrected_000/BaseCalled_template/"].attrs:
                        read.attrs[k] = tmp["Analyses/RawGenomeCorrected_000/BaseCalled_template/"].attrs[k]
                except KeyError:
                    error_template += 1
                    continue

                try:
                    al = read.create_group("Alignment")
                except ValueError:
                    al = read["Alignment"]
                    pass
                try:
                    for k in tmp["Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment"].attrs:
                        al.attrs[k] = tmp["Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment"].attrs[k]
                except KeyError:
                    error_al += 1
                    continue
                tmp_e = tmp["Analyses/RawGenomeCorrected_000/BaseCalled_template/Events"]
                try:
                    event = read.create_dataset(
                        "Event", tmp_e.shape, dtype=tmp_e.dtype, data=tmp_e.value)
                except:
                    pass
    print("Nerror al", error_al)
    print("Nerror template", error_template)
