import torch
import numpy as np

class RNNLM(torch.nn.Module):
    def __init__(self,n_vocab, d_embed, d_hidden, n_layers, dropout_rate):
        super(RNNLM, self).__init__()
        self.n_vocab= n_vocab
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        
        self.embed = torch.nn.Embedding(n_vocab, d_embed)
        self.rnn = torch.nn.GRU(d_embed, d_hidden,n_layers, batch_first=True, dropout=dropout_rate)
        self.proj = torch.nn.Linear(d_hidden, n_vocab)
        
    def forward(self, x):
        embed_x = self.embed(x)
        hs,_ = self.rnn(embed_x)
        out = self.proj(hs)
        return out
    
    def sampling(self, lang, start_words="", max_len=50):
        self.eval()
        sos = lang.vocab["<sos>"]
        eos = lang.vocab["<eos>"]

        hypos = []
        next_word = sos if not start_words else start_words
        while next_word!= eos or len(hypos)<max_len:
            hypos.append(next_word)
            inputs = torch.LongTensor(hypos)
            inputs = inputs.unsqueeze(0)
            inputs= inputs.cuda()
            outputs = self.forward(inputs)
            probs = torch.softmax(outputs[0][-1],dim=0).cpu().detach().numpy()
            next_word = np.random.choice(range(len(probs)), p=probs)
            
        
        return lang.idx2str(hypos)