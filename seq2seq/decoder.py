import mxnet as mx
from mxnet import gluon, nd, autograd
import numpy as np


class Decoder(gluon.Block):
    def __init__(self, vocab_size, n_embed, n_hidden, n_proj):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        with self.name_scope():
            self.embed = gluon.nn.Embedding(vocab_size, n_embed)
            self.rnn = gluon.rnn.LSTMCell(n_hidden, input_size=n_embed)
            self.proj = gluon.nn.Dense(
                n_proj, in_units=n_hidden, activation="relu")
            self.out = gluon.nn.Dense(vocab_size, in_units=n_proj)

    def forward(self, X, states=None, ctx=mx.cpu()):
        xs_embedding = self.embed(X)

        n_state = 2 if isinstance(
            self.rnn, (gluon.rnn.LSTM, gluon.rnn.LSTMCell)) else 1
        if states is None:
            states = [nd.zeros(shape=[X.shape[0], self.n_hidden], ctx=ctx)
                      for i in range(n_state)]
        else:
            states = [states[i][:X.shape[0]] for i in range(n_state)]

        y, states = self.rnn(xs_embedding, states)
        out = self.out(self.proj(y))

        return out, states

    def inference(self, token, states, ctx=mx.cpu()):
        '''
            Infer the next token given the previous token and hidden states
            -----
            Input:

        '''
        token = nd.array(token).as_in_context(ctx)
        x_embedding = self.embed(token)
        y, states = self.rnn(x_embedding, states)
        out = nd.log_softmax(self.out(self.proj(y)))
        return out, states