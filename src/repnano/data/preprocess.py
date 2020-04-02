from tombo import tombo_helper, tombo_stats, resquiggle
from tqdm import tqdm
import numpy as np
import mappy
import h5py
import io
from joblib import Parallel, delayed
import copy


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



"""
rsqgl_res = resquiggle_read(
            map_res, std_ref, params, outlier_thresh,
            const_scale=const_scale, skip_seq_scaling=skip_seq_scaling,
            seq_samp_type=seq_samp_type)
        n_iters = 1
        #print("maxs,",max_scaling_iters)
        while n_iters < max_scaling_iters and rsqgl_res.norm_params_changed:

            rsqgl_res = resquiggle_read(
                map_res._replace(scale_values=rsqgl_res.scale_values),
                std_ref, params, outlier_thresh, all_raw_signal=all_raw_signal,
                seq_samp_type=seq_samp_type)
            n_iters += 1
        return rsqgl_res

"""


def process_h5(fast5_data, aligner):
    seq_samp_type = tombo_helper.seqSampleType("DNA", False)  # Impose DNA  but what is rev_sig
    # prep aligner, signal model and parameters

    std_ref = tombo_stats.TomboModel(seq_samp_type=seq_samp_type)
    rsqgl_params = tombo_stats.load_resquiggle_parameters(seq_samp_type)

    # extract data from FAST5
    map_results = resquiggle.map_read(fast5_data, aligner,
                                      std_ref)  # Should be modified at that point to insert sequence from fasta
    #print(map_results)
    all_raw_signal = tombo_helper.get_raw_read_slot(fast5_data)['Signal'][:]
    if seq_samp_type.rev_sig:
        all_raw_signal = all_raw_signal[::-1]
    map_results = map_results._replace(raw_signal=all_raw_signal)

    # run full re-squiggle

    #use_save_bandwidth = True
    try:

        rsqgl_res = resquiggle.resquiggle_read(
            map_results, std_ref, rsqgl_params, all_raw_signal=all_raw_signal,outlier_thresh=5)

    #cprs = copy.deepcopy(rsqgl_params)
    #cprs.use_save_bandwidth = True
    except:
        cprs = tombo_stats.load_resquiggle_parameters(
            seq_samp_type, use_save_bandwidth=True)
        #print(cprs)
        rsqgl_res = resquiggle.resquiggle_read(
            map_results, std_ref, cprs, all_raw_signal=all_raw_signal, outlier_thresh=5)
    #except:
    #    pass
    """
       """

    """
    n_iters = 1
    max_scaling_iters = 3
    while n_iters < max_scaling_iters and rsqgl_res.norm_params_changed:
        rsqgl_res = resquiggle.resquiggle_read(
            map_results._replace(scale_values=rsqgl_res.scale_values),
            std_ref, rsqgl_params, outlier_thresh=5, all_raw_signal=all_raw_signal,
            seq_samp_type=seq_samp_type)
        n_iters += 1
    """
    norm_means = tombo_helper.c_new_means(rsqgl_res.raw_signal, rsqgl_res.segs)
    norm_stds = tombo_helper.repeat(np.NAN)

    event_data = np.array(
        list(zip(np.array(norm_means, dtype=np.float16),
                 list(rsqgl_res.genome_seq))),
        dtype=[(str('norm_mean'), np.float16),
               (str('base'), 'S1')])

    return event_data ,rsqgl_res


def create_event(h5p, event_data,rsqgl_res):
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

    corr_alignment = BC['Alignment']
    corr_alignment.attrs['mapped_start'] = rsqgl_res.genome_loc.Start
    corr_alignment.attrs[
        'mapped_end'] = rsqgl_res.genome_loc.Start + len(rsqgl_res.segs) - 1
    corr_alignment.attrs[
        'mapped_strand'] = rsqgl_res.genome_loc.Strand
    corr_alignment.attrs['mapped_chrom'] = rsqgl_res.genome_loc.Chrom

    if rsqgl_res.align_info is not None:
        corr_alignment.attrs[
            'clipped_bases_start'] = rsqgl_res.align_info.ClipStart
        corr_alignment.attrs[
            'clipped_bases_end'] = rsqgl_res.align_info.ClipEnd
        corr_alignment.attrs[
            'num_insertions'] = rsqgl_res.align_info.Insertions
        corr_alignment.attrs[
            'num_deletions'] = rsqgl_res.align_info.Deletions
        corr_alignment.attrs[
            'num_matches'] = rsqgl_res.align_info.Matches
        corr_alignment.attrs[
            'num_mismatches'] = rsqgl_res.align_info.Mismatches




def process_one_big_hdf5(hdf5_name, fn_fastq, ref, output_name,njobs,maxlen=None):
    error = {"seq_not_found": 0}
    if ref is not None:
        Al = mappy.Aligner(ref, preset="map-ont")

    # First one run over the keys to get the name that we will need for the fastq:
    # We do this to decrease memory usage
    key_to_keep = []
    with h5py.File(hdf5_name, "r") as h5:
        keys = list(h5.keys())

        print("Total number of reads %i" % len(keys))
        for ik, k in enumerate(keys):
            # print(k)
            h5p = h5[k]
            # print(h5p)
            key_to_keep.append(get_name(h5p))

    # Get fasta seq
    data_fastq = {}
    for name, seq, qual, comment in mappy.fastx_read(fn=fn_fastq,
                                                     read_comment=True):
        # print(name,seq,qual,comment)
        name_fast = "_".join(name.split("_")[:2])
        if name_fast in key_to_keep:
            data_fastq[name_fast] = "\n".join([name, seq, "None", qual])  # WARRRRRRRRNNNINGGGGGGGGGG

    n_processed = 0

    def copy_raw(raw_source, destination):
        destination.create_group("Raw")
        raw = destination["Raw"]

        # for el in ["Raw"]:#,"context_tags","tracking_id"]:
        for k in raw_source.attrs:
            # print("k",k,h5p[el].attrs[k])
            raw.attrs[k] = raw_source.attrs[k]
        # raw.attrs["read_id"] = h5p["Raw"].attrs["read_id"]
        read = raw.create_group("Reads")
        readn = int(raw_source.attrs["read_number"])

        read2 = read.create_group("Read_%i" % readn)
        for k in raw_source.attrs:
            # print("k",k,h5p[el].attrs[k])
            read2.attrs[k] = raw_source.attrs[k]
        shape = raw_source["Signal"].shape
        # print(np.median(h5p["Raw/Signal"].value),len(h5p["Raw/Signal"].value))
        read2.create_dataset("Signal", shape, dtype="<i2", data=raw_source["Signal"].value)

    def create_virtual(fasta, raw):
        virtual = io.BytesIO()
        error = {}
        with h5py.File(virtual) as h5:
            assign_fasta(h5, fasta)
            copy_raw(raw_source=raw, destination=h5)

        return virtual

    def virtual_h5_to_processing(virtual,ref):

        if ref is not None:
            if type(ref) == str:
                #print("Loading allign, virtual")
                Al = mappy.Aligner(ref, preset="map-ont")
            else:
                Al = ref

        error = {}
        event_data =[]
        resq_res = []
        with h5py.File(virtual,"w") as h5:

            try:
                event_data,resq_res = process_h5(h5, aligner=Al)

            except tombo_helper.TomboError as err:
                if len(err.args) > 0:
                    msg = err.args[0]

                error[msg] = 1

            return event_data,resq_res, error

    def process_chunk(virtuals,ref):

        if ref is not None:
            if type(ref) == str:
                #print("Loading allign, chunks")
                Al = mappy.Aligner(ref, preset="map-ont")
            else:
                Al = ref
        res = {}
        for virtual,k in virtuals:
            if virtual is not None:
                res[k] = virtual_h5_to_processing(virtual,Al)
        return res

    # To do multiprocess first create virtuals then process
    data = {}
    data = {}

    from itertools import zip_longest  # for Python 3.x
    def grouper(n, iterable, padvalue=None):
        "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
        return list(zip_longest(*[iter(iterable)] * n, fillvalue=padvalue))

    with h5py.File(hdf5_name, "r+") as h5:
        keys = list(h5.keys())

        #Create virtual file by batch of 400

        for group_k in grouper(400,keys[:maxlen]):
            print("create virtuals by batch of 400")

            list_k = []
            virtuals = {}

            for ik, k in enumerate(group_k):
                # print(k)

                if k is None:
                    continue

                h5p = h5[k]
                # print(h5p)
                name = get_name(h5p)
                if name not in data_fastq:
                    error["seq_not_found"] += 1
                    continue

                virtuals[k] = create_virtual(data_fastq[name], raw=h5[k]["Raw"])
                list_k.append(k)


            if njobs == 1:
                for k in tqdm(list_k):
                    data[k] = virtual_h5_to_processing(virtuals[k], Al)
            else:
                # from six.moves import zip_longest # for both (uses the six compat library)


                print("Process by chunk  of 20")
                #create chunk of about 20
                #split = int(len(list_k) / (10))
                list_virtual_lists = grouper(20,[[virtuals[k],k] for k in list_k],None)
                #print(list_virtual_lists)
                gdata = Parallel(n_jobs=njobs)(delayed(process_chunk)(virtuals_l,ref) for virtuals_l in tqdm(list_virtual_lists))
                #print(gdata)
                for sublist in gdata :
                    for k,item in sublist.items():
                        data[k] = item
                del gdata

        #print(data)
    #
    # sk(h5p["Analyses/RawGenomeCorrected_000/BaseCalled_template/"])

    # Events goes there

    # Then write
    with h5py.File(output_name, "w") as h5:
        for k, (events,rsq_res, err) in data.items():
            if len(err) == 0:
                h5.create_group(k)
                create_event(h5[k], events,rsq_res)
                n_processed += 1
            else:
                errm = list(err.keys())[0]
                if errm in error.keys():
                    error[errm] += 1
                else:
                    error[errm] = 1

    print("\nTotal errors", np.sum([v for v in error.values()]))
    print("detail of errors:")
    print(error)

    print("\nSucefully processed", n_processed)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5', type=str)
    parser.add_argument('--fastq', type=str)
    parser.add_argument('--ref', type=str)
    parser.add_argument('--output_name', type=str)
    parser.add_argument('--njobs', type=int,default=1)
    parser.add_argument('--maxlen', type=int,default=None)



    args = parser.parse_args()

    process_one_big_hdf5(hdf5_name=args.hdf5,fn_fastq=args.fastq,
                         ref=args.ref,output_name=args.output_name,njobs=args.njobs,maxlen=args.maxlen)

#{'seq_not_found': 0, 'Read event to sequence alignment extends beyond bandwidth': 1003, 'Alignment not produced': 415}
