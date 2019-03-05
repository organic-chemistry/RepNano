if __name__ == "__main__":
    import argparse

    import os
    import numpy as np
    # rom subprocess import PIPE
    # OMP_NUM_THREADS=1 THEANO_FLAGS=mode=FAST_RUN
    parser = argparse.ArgumentParser()
    parser.add_argument('--command-line', dest="command_line", type=str)
    parser.add_argument("--n-cpu", dest="n_cpu", type=int, default=None)
    parser.add_argument("--range", dest="range", nargs='+', default=[], type=float)
    parser.add_argument("--name", dest="name", type=str, default='dataset.pick')

    args = parser.parse_args()

    os.makedirs(os.path.split(args.name)[1], exist_ok=True)

    for i in range(args.n_cpu):
        start = np.arange(args.range[0], args.range[1] + 1e-4,
                          (args.range[1] - args.range[0]) / args.n_cpu)[i]
        end = np.arange(args.range[0], args.range[1] + 1e-4,
                        (args.range[1] - args.range[0]) / args.n_cpu)[i + 1]

        cmd = args.command_line + \
            "--range %.3f %.3f " % (start, end) + "--name %s-%.3f " % (args.name, start)
        print(cmd)
        os.popen(cmd)
