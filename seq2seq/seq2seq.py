import mxnet as mx
from mxnet import gluon, nd, autograd
import numpy as np

class Seq2SeqModel(gluon.Block):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, Xs, Ys, states=None, ctx=mx.cpu()):
        '''
           Gererate output given inputs
           -----
           Input:
           -----
           Output:
           *(nd.narray): output in each timestep, shape:(batch_size * timestep, vocab_size)
        '''
        # Encoder
        for X in Xs:
            y, states = self.encoder.forward(X, states=states, ctx=ctx)

        # Decoder
        outs = []
        for Y in Ys:
            out, states = self.decoder.forward(Y, states=states, ctx=ctx)
            outs.append(out)

        return nd.concat(*outs, dim=0)

    def generate(self, Xs, max_len, states=None, method=None, start="<s>", end="</s>", ctx=mx.cpu()):
        '''
            Generate translation given source sentence.
            -----
            Input:
            -----
            Output:
        '''
        # encode step
        for X in Xs:
            y, states = self.encoder.forward(X, states, ctx)

        # decode step
        hypo_sents = [start]
        for i in range(max_len+1):
            out, states = self.decoder.inference(hypo_sents[i], states, ctx=ctx)
            hypo = nd.argmax(out, axis=1)
            hypo_sents.append(hypo)

            if hypo.asscalar() == end:
                break
        return nd.concat(*hypo_sents, dim=0)[1:]