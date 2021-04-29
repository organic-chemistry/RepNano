from tombo import tombo_helper, tombo_stats, resquiggle
from tqdm import tqdm
import numpy as np
import mappy
import h5py
import io
from joblib import Parallel, delayed
import errno
import os
import glob
import copy


def get_names(h5p):
    ch = int(str(h5p["channel_id"].attrs["channel_number"])[2:-1])
    readn = int(h5p["Raw"].attrs["read_number"])
    #print(h5p["Raw"].attrs["read_id"])
    #print()
    return "ch%i_read%i" % (ch, readn) , h5p["Raw"].attrs["read_id"].decode("utf-8")


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

    #print(rsqgl_params)
    #for k in std_ref.sds.keys():
    #    if k[2:3] == "T" or k[3:4] == "T":
    #        #pass
    #        std_ref.sds[k] = 1.
    #print(std_ref.sds)
    # extract data from FAST5
    map_results = resquiggle.map_read(fast5_data, aligner,
                                      std_ref)  # Should be modified at that point to insert sequence from fasta
    #print(map_results)
    bases = set(map_results.genome_seq)
    non_canonical = [b for b in bases if b not in ["a","t","c","g","A","T","C","G"]]
    if len(non_canonical) != 0:
        raise ValueError("Found non canonical bases (%s)"%str(non_canonical))

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



def read_fastq(fastq,to_keep):
    data_fastq = {}
    with open(fastq,"r") as f:
        data = []
        keep = False
        line  = f.readline()
        while line:
            #print(line)
            if line.startswith("@"):
                if len(data) == 4:
                    name_fast = "_".join(data[0].split("_")[:2])
                    data[0] =data[0]
                    data[2]="None"
                    #data[-1] = data[-1].replace("\\'","'")
                    data_fastq[name_fast] = "\n".join(data)
                    #print(data)
                    #print(len(data[-1]),type(data[-1]))

                line = line[1:]
                name_fast = "_".join(line[:-1].split("_")[:2])
                #print(line)
                #print(name_fast,len(data))
                if name_fast in to_keep:
                    keep = True

                else:
                    keep = False
                data = []
            if keep:
                data.append(u"""%s"""%line[:-1])

            line = f.readline()
    #print(to_keep)
    #print("size",len(data_fastq))
    return data_fastq


def process_one_big_hdf5(hdf5_name, fn_fastq, ref, output_name,njobs,maxlen=None,fastqs=None):
    error = {"seq_not_found": 0}

    if ref is not None:
        if not os.path.exists(ref):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), ref)

        Al = mappy.Aligner(ref, preset="map-ont")

    # First one run over the keys to get the name that we will need for the fastq:
    # We do this to decrease memory usage
    key_to_keep1 = []
    key_to_keep2 = []
    with h5py.File(hdf5_name, "r") as h5:
        keys = list(h5.keys())

        print("Total number of reads %i" % len(keys))
        for ik, k in enumerate(keys):
            # print(k)
            h5p = h5[k]
            # print(h5p)
            id1,id2 = get_names(h5p) #depend of the formating
            key_to_keep1.append(id1)
            key_to_keep2.append(id2)

            #print(id1,id2)

    # Get fasta seq
    data_fastq = {}
    if fastqs == None:
        fastqs = [fn_fastq]
    for fastq in fastqs:
        print("Reading",fastq)

        for name, seq, qual, comment in mappy.fastx_read(fn=fastq,
                                                         read_comment=True):
            #print(name,seq,qual,comment)
            
            name_fast = "_".join(name.split("_")[:2])


            read_id = name.split()[0].split('_')[0]

            #print(name_fast, read_id)

            if name_fast in key_to_keep1 or read_id in key_to_keep2:
                #print(name_fast,read_id)
                if read_id in key_to_keep2:
                    name_fast = read_id
                #print(name,seq,qual)
                data_fastq[name_fast] = "\n".join([name, seq, "None", qual])  # WARRRRRRRRNNNINGGGGGGGGGG
                #print(len(qual), type(qual))

        #data_fastq.update(read_fastq(fastq,key_to_keep))
    print("si",len(data_fastq))
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
        read2.create_dataset("Signal", shape, dtype="<i2", data=raw_source["Signal"])

    def create_virtual(fasta, raw):
        virtual = io.BytesIO()
        error = {}
        with h5py.File(virtual,"w") as h5:
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

            except (tombo_helper.TomboError,ValueError) as err:
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
        if virtuals is None or len(virtuals) == 0:
            return res
        #print(virtuals)
        for block in virtuals:
            if block is None:
                continue
            virtual, k = block
            if virtual is not None:
                try:
                    res[k] = virtual_h5_to_processing(virtual,Al)
                except IndexError as err:
                    error = {}
                    if len(err.args) > 0:
                        msg = err.args[0]
                        error[msg] = 1
                    else:
                        error["IndexError"] += 1
                    res[k] =[ [] ,[] , error]


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
                name1,name2 = get_names(h5p)
                name = name1
                if name1 not in data_fastq:
                    name = name2
                    if name2 not in data_fastq:
                        error["seq_not_found"] += 1
                        continue

                #print(name1,name2)

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

    print("\nSuccessfully processed", n_processed)

    for k in error.keys():
        if "Found non canonical bases" in k:

            print("\n\nNon canonical bases error can be resolved by changing the non canonical bases by canonical bases in the reference genome")
            print("For example sed 's/N/A/g' ref_with_N.txt > ref_corrected.txt")
            print("To replace all N bases by A bases")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5', type=str)
    parser.add_argument('--fastq', type=str)
    parser.add_argument('--fastqs', type=str,nargs='+',default=None)
    parser.add_argument('--ref', type=str)
    parser.add_argument('--output_name', type=str)
    parser.add_argument('--njobs', type=int,default=1)
    parser.add_argument('--max_len', type=int,default=None)
    parser.add_argument('--Nfile', type=int,default=1)



    args = parser.parse_args()
    dire = os.path.split(args.output_name)[0]
    if dire != "":
        os.makedirs(dire, exist_ok=True)
    if args.Nfile == 1:
        process_one_big_hdf5(hdf5_name=args.hdf5,fn_fastq=args.fastq,
                         ref=args.ref,output_name=args.output_name,njobs=args.njobs,
                             maxlen=args.max_len,fastqs=args.fastqs)

    else:
        ls = glob.glob(args.hdf5+"/*.fast5")
        ls.sort()
        if args.Nfile != 0:
            ls = ls[:args.Nfile]
        for i,fast5 in enumerate(ls):
            print("Processing",fast5)
            process_one_big_hdf5(hdf5_name=fast5, fn_fastq=args.fastq,
                                 ref=args.ref, output_name=args.output_name +f"/output_{i}.h5", njobs=args.njobs,
                                 maxlen=args.max_len, fastqs=args.fastqs)

#{'seq_not_found': 0, 'Read event to sequence alignment extends beyond bandwidth': 1003, 'Alignment not produced': 415}
