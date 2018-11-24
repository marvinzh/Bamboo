import argparse
import logging
import mxnet as mx
from mxnet import gluon, nd, autograd
import numpy as np

from util import dataset
import seq2seq
import encoder
import decoder


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training a seq2seq model")

    # dataset-related settings
    parser.add_argument("--source", help="source file path for training")
    parser.add_argument("--target", help="target file path for training")
    # parser.add_argument(
    #     "--valid-source", help="source file path for validation")
    # parser.add_argument(
    #     "--valid-target", help="target file path for validation")

    # Encoder-related settings
    parser.add_argument("--enc-vocab", type=int, default=10000,
                        help="vocabulary size for encoder")
    parser.add_argument("--enc-embed", type=int, default=64,
                        help="embedding size for encoder")
    parser.add_argument("--enc-hidden", type=int, default=128,
                        help="hidden size for encoder")

    # Decoder-related settings
    parser.add_argument("--dec-vocab", type=int, default=10000,
                        help="vocabulary size for decoder")
    parser.add_argument("--dec-embed", type=int, default=64,
                        help="embedding size for decoder")
    parser.add_argument("--dec-hidden", type=int, default=128,
                        help="hidden size for decoder")
    parser.add_argument("--dec-proj", type=int, default=128,
                        help="projection size for decoder")

    # training-related settings
    parser.add_argument("--batch-size", type=int, default=64,
                        help="batch size for training the model")
    parser.add_argument("--epoches", type=int, default=10,
                        help="max epoches to train the model")
    parser.add_argument("--n-gpu", type=int, default=0,
                        help="# of GPU to use, 0 or negative number denotes use CPU")
    # parset.add_argument("--opt",help="Optimizer")
    # parser.add_argument("--clip-norm", int, help="norm for clip gradient")
    # parser.add_argument("--snapshot")

    # other settings
    parser.add_argument("--log", help="output log file path")
    # parser.add_argument("--keep-n",type=int ,help="only keep n best file in the training process")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    # set logger setting
    logging.basicConfig(filename=args.log,
                        level=logging.INFO,
                        format="%(asctime)s [%(levelname)s]: %(message)s",
                        # datefmt="%m-%d %H:%M:%S"
                        )

    if args.log is not None:
        # redirect output steam to screen
        console = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
        console.setFormatter(formatter)
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    logging.info("Training args:%s"%args)

    data = dataset.generate_TS_data(
        [10, 13, 64], sos=9998, eos=9999, ctx=mx.cpu())
    # print("Sanity Check:")
    # print("# of batch: ", len(data))
    # print("shape of X: ", data[0][0].shape, type(data[0][0]))
    # print("shape of Y: ", data[0][1].shape, type(data[0][1]))

    # infer some parameters
    context = mx.cpu() if args.n_gpu < 1 else mx.gpu(args.n_gpu)
    num_batch = len(data)
    timesteps = 13

    model = seq2seq.Seq2SeqModel(
        encoder.Encoder(args.enc_vocab, args.enc_embed, args.enc_hidden),
        decoder.Decoder(args.dec_vocab, args.dec_embed,
                        args.dec_hidden, args.dec_proj)
    )

    # initialize model parameters
    model.collect_params().initialize(mx.init.Xavier(), ctx=context)
    # calculate the loss function
    loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True)
    # set optimizer
    optimizer = gluon.Trainer(model.collect_params(), mx.optimizer.Adam())

    costs = []
    for epoch in range(args.epoches):
        cum_loss = 0.
        log_probs = 0.
        data_idx = np.random.permutation(len(data))
        for idx in data_idx:
            X = data[idx][0].as_in_context(context)
            Y = data[idx][1].as_in_context(context)
            with mx.autograd.record():
                out = model.forward(X, Y[:-1], ctx=context)
                losses = loss(out, Y[1:].reshape(shape=[-1]))
                losses.backward()
                cum_loss += losses.mean().asscalar()

            grads = [i.grad(context) for i in model.collect_params().values()]

            gluon.utils.clip_global_norm(grads, 1)
            optimizer.step(args.batch_size)
            model.collect_params().zero_grad()

            idx_mask = nd.one_hot(Y[1:].reshape(shape=[-1]), 10000)
            log_probs += nd.sum(nd.log_softmax(out) * idx_mask).asscalar()

        cost = cum_loss / num_batch
        costs.append(cost)
        logging.info("Epoch: %d/%d, cost: %f, Perplexity: %f" % (epoch, args.epoches,
                                                                 cost, np.exp(-log_probs / (num_batch*timesteps * args.batch_size))))
