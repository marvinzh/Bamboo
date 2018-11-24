from mxnet import ndarray as nd


def generate_TS_data(shape, sos, eos, ctx):
    
    """randomly generating time-series data

    Arguments:
        shape {list-like} -- (batches,time_step, data) 
        sos {int} -- idx denotes start of the sequence
        eos {int} -- idx denotes end of the sequence
        ctx {mx.cpu() or mx.gpu()} -- device holding the data
    """
    assert len(shape)==3

    N, T, B = shape

    dataset = []
    for i in range(N):
        Xs = nd.floor(
            nd.random_uniform(low=0, high=9998, shape=[T, B], ctx=ctx)
        )

        Ys = Xs.copy()
        end = nd.ones(shape=[1, B], ctx=ctx) * 9998
        sos = nd.ones(shape=[1, B], ctx=ctx) * 9999
        Xs = nd.concat(Xs, end, dim=0)
        Ys = nd.concat(sos, Ys, end, dim=0)
        dataset.append(
            (Xs, Ys)
        )

    return dataset
