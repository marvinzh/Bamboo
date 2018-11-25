from mxnet import ndarray as nd
import pickle
import argparse
import mxnet as mx


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Genarate toy date for testing")

    parser.add_argument("--output", help="path for output file")
    parser.add_argument("--shape", type=int, nargs="+",
                        help="shape for generated data")
    parser.add_argument("--sos", type=int, default=0,
                        help="start of the sequence")
    parser.add_argument("--eos", type=int, default=1,
                        help="end of the sequence")

    parser.add_argument("--n-gpu", type=int, default=0,
                        help="# of GPU to use, 0 or negative number denotes use CPU")

    return parser.parse_args()


def generate_TS_data(shape, ctx):
    """randomly generating time-series data

    Arguments:
        shape {iterable} -- (#batches, time_step, data) 
        sos {int} -- idx denotes start of the sequence
        eos {int} -- idx denotes end of the sequence
        ctx {mx.cpu() or mx.gpu()} -- device holding the data
    """
    assert len(shape) == 3

    dataset = nd.random_uniform(low=3, high=9999, shape=shape, ctx=ctx)

    return dataset


def save(file, obj):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":
    args = parse_arguments()
    context = mx.cpu() if args.n_gpu < 1 else mx.gpu(args.n_gpu)

    data = generate_TS_data(args.shape, context)

    save(args.output, data)
