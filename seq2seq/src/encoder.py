import mxnet as mx
from mxnet import gluon, nd, autograd
import numpy as np


class Encoder(gluon.Block):
    def __init__(self, vocab_size, n_embed, n_hidden):
        '''
            vocab_size (int): vocabulary size
            n_embed (int): embedding size
            n_hidden (int): number of hidden units
        '''
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        with self.name_scope():
            self.embed = gluon.nn.Embedding(vocab_size, n_embed)
            self.rnn = gluon.rnn.LSTMCell(n_hidden, input_size=n_embed)

    def forward(self, X, states=None, ctx=mx.cpu()):
        '''
            Forward propagation of Encoder.
            -----
            Input:
            X(nd.array): inputs (word index) for current timestep. in shape: (batchsize,).
            states: list of hidden states for current timestep. [h, c] if use LSTM else [h].
            ctx: model context.
            -----
            Output:
            y: hidden state of current timestep
            states: list of hidden state of current timestep, [h, c] if use LSTM else [h].
        '''
        xs_embedding = self.embed(X)

        n_state = 2 if isinstance(
            self.rnn, (gluon.rnn.LSTM, gluon.rnn.LSTMCell)) else 1

        if states is None:
            states = [nd.zeros(shape=[X.shape[0], self.n_hidden], ctx=ctx)
                      for i in range(n_state)]
        else:
            states = [states[i][:X.shape[0]] for i in range(n_state)]

        y, states = self.rnn(xs_embedding, states)
        
        return y, states
