from tombo import tombo_helper, tombo_stats, resquiggle
from tqdm import tqdm
import numpy as np
import mappy
import h5py


def get_name(h5p):
    ch = int(str(h5p["channel_id"].attrs["channel_number"])[2:-1])
    readn = int(h5p["Raw"].attrs["read_number"])
    return "ch%i_read%i" % (ch, readn)


def assign_fasta(h5p, fasta):
    # print(name)
    # break

    if "Analyses" not in h5p.keys():
        h5p.create_group("Analyses")
    Ana = h5p["Analyses"]

    if "Basecall_1D_000" not in Ana.keys():
        Ana.create_group("Basecall_1D_000")
    Temp = Ana["Basecall_1D_000"]
    if 'BaseCalled_template' not in Temp.keys():
        Temp.create_group('BaseCalled_template')
    Temp = Temp['BaseCalled_template']
    if 'Fastq' not in Temp.keys():
        pass
    else:
        # print("Delet")
        del Temp["Fastq"]

    # print(Temp["Fastq"].value)
    Temp.create_dataset('Fastq', data=fasta,
                        dtype=h5py.special_dtype(vlen=str))


def process_h5(fast5_data, aligner):
    seq_samp_type = tombo_helper.seqSampleType("DNA", False)  # Impose DNA  but what is rev_sig
    # prep aligner, signal model and parameters

    std_ref = tombo_stats.TomboModel(seq_samp_type=seq_samp_type)
    rsqgl_params = tombo_stats.load_resquiggle_parameters(seq_samp_type)

    # extract data from FAST5
    map_results = resquiggle.map_read(fast5_data, aligner,
                                      std_ref)  # Should be modified at that point to insert sequence from fasta
    # print(map_results)
    all_raw_signal = tombo_helper.get_raw_read_slot(fast5_data)['Signal'][:]
    if seq_samp_type.rev_sig:
        all_raw_signal = all_raw_signal[::-1]
    map_results = map_results._replace(raw_signal=all_raw_signal)

    # run full re-squiggle
    rsqgl_res = resquiggle.resquiggle_read(
        map_results, std_ref, rsqgl_params, all_raw_signal=all_raw_signal)

    norm_means = tombo_helper.c_new_means(rsqgl_res.raw_signal, rsqgl_res.segs)
    norm_stds = tombo_helper.repeat(np.NAN)

    event_data = np.array(
        list(zip(norm_means, norm_stds,
                 rsqgl_res.segs[:-1], np.diff(rsqgl_res.segs),
                 list(rsqgl_res.genome_seq))),
        dtype=[(str('norm_mean'), 'f8'), (str('norm_stdev'), 'f8'),
               (str('start'), 'u4'), (str('length'), 'u4'),
               (str('base'), 'S1')])

    # "
    return event_data


def soft_link_raw(h5p, k):
    Raw = h5p["Raw"]

    if "Reads" not in Raw.keys():
        Raw.create_group("Reads")
    readn = int(Raw.attrs["read_number"])

    Reads = Raw["Reads"]
    key = "Reads_%i" % readn

    if "Signal" in Reads.keys():
        del Reads["Signal"]

    if key not in Reads:
        Reads.create_group(key)

    Reads2 = Reads[key]

    Reads2.attrs["read_id"] = h5p["Raw"].attrs["read_id"]
    if "Signal" not in Reads2.keys():
        Reads2["Signal"] = h5py.SoftLink('/%s/Raw/Signal' % k)

    # print(Reads2)
    # print(Reads2["Signal"].value)


def create_event(h5p, event_data):
    if "BaseCalled_template" not in h5p.keys():
        h5p.create_group("BaseCalled_template")
    BC = h5p["BaseCalled_template"]

    """
    if "RawGenomeCorrected_000" not in Ana.keys():
        Ana.create_group("RawGenomeCorrected_000")
    Temp =  Ana["RawGenomeCorrected_000"]
    if 'BaseCalled_template' not in Temp.keys():
        Temp.create_group('BaseCalled_template')
    Temp = Temp['BaseCalled_template']
    """

    if "Events" not in BC.keys():
        corr_events = BC.create_dataset(
            'Events', data=event_data, compression="gzip")
    if "Alignment" not in BC.keys():
        BC.create_group("Alignment")
    """
    corr_events.attrs[
            'read_start_rel_to_raw'] = rsqgl_res.read_start_rel_to_raw"""


def process_one_big_hdf5(hdf5_name, fn_fastq, ref):
    error = {"seq_not_found": 0}
    if ref is not None:
        Al = mappy.Aligner(ref, preset="map-ont")

    data_fastq = {}
    for name, seq, qual, comment in mappy.fastx_read(fn=fn_fastq,
                                                     read_comment=True):
        # print(name,seq,qual,comment)
        name_fast = "_".join(name.split("_")[:2])
        data_fastq[name_fast] = "\n".join([name, seq, "None", qual])  # WARRRRRRRRNNNINGGGGGGGGGG
        # break
    # return
    n_processed = 0
    with h5py.File(hdf5_name, "r+") as h5:
        keys = list(h5.keys())

        for pending in ["Analyses", "Raw"]:
            if pending in keys:
                keys.remove(pending)
                del h5["/" + pending]
        print("Total number of reads %i" % len(keys))
        for ik, k in tqdm(enumerate(keys)):
            # print(k)
            h5p = h5[k]
            # print(h5p)
            name = get_name(h5p)
            if name not in data_fastq:
                error["seq_not_found"] += 1
                continue

            fasta = data_fastq[name]
            assign_fasta(h5p, fasta)


            #Create sof links. Only needed for bigf
            h5["/Analyses"] = h5py.SoftLink('/%s/Analyses' % k)
            # create soft links in raw to signal
            soft_link_raw(h5p, k)

            h5["/Raw"] = h5py.SoftLink('/%s/Raw' % k)

            try:
                event_data = process_h5(h5p, aligner=Al)

            except tombo_helper.TomboError as err:
                if len(err.args) > 0:
                    msg = err.args[0]
                if msg in error.keys():
                    error[msg] += 1
                else:
                    error[msg] = 1
                # print(err,err.args )
                # print("Er")
                #error["process_error"] += 1
                del h5["/Analyses"]
                del h5["/Raw"]
                continue

            del h5["/Analyses"]
            del h5["/Raw"]
            create_event(h5p, event_data)
            n_processed += 1
            # sk(h5p["Analyses/RawGenomeCorrected_000/BaseCalled_template/"])

        # Events goes there
    print("\nTotal errors",np.sum([v for v in error.values()]))
    print("detail of errors:")
    print(error)

    print("\nSucefully processed",n_processed)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5', type=str)
    parser.add_argument('--fastq', type=str)
    parser.add_argument('--ref', type=str)


    args = parser.parse_args()

    process_one_big_hdf5(hdf5_name=args.hdf5,fn_fastq=args.fastq,ref=args.ref)
