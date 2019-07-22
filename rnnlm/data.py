import torch
from torch.utils.data import DataLoader, Dataset

def word_tokenize(corpus):
    char_tokens = []
    for line in corpus:
        char_tokens.extend(list(line))
    
    return char_tokens

def padding_for_poetry(batch,padding_value):
    padding = lambda x:torch.nn.utils.rnn.pad_sequence(x, batch_first=True,padding_value=padding_value)
     
    xs, ys = zip(*batch)
    return padding(xs), padding(ys)

class Lang:
    def __init__(self, corpus, reserved_tokens=[], n_vocab=-1):
        print()
        vocab_list = ["<unk>","<pad>","<sos>","<eos>"]+reserved_tokens + self._build_vocab(corpus)[::-1]
        vocab_list = vocab_list[:n_vocab] if n_vocab > 0 else vocab_list

        self.vocab_list = vocab_list
        self.vocab = dict()
        for i, token in enumerate(vocab_list):
            self.vocab[token] = i

    def __len__(self):
        return len(self.vocab)
    
    @property
    def size(self):
        return len(self.vocab)

    def idx2str(self, idx_seq):
        return list(map(
            lambda i: self.vocab_list[i], idx_seq
        ))

    def str2idx(self, str_seq):
        return list(map(
            lambda k: self.vocab[k] if k in self.vocab else self.vocab["<unk>"], str_seq
        ))

    def add(self, token):
        self.vocab[token] = len(self.vocab)
        self.vocab_list.append(token)

    def _build_vocab(self, corpus):
        vocab = dict()
        tokens = word_tokenize(corpus)
        for token in tokens:
            if token not in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1


        vocab_list = [(key, vocab[key]) for key in vocab.keys()]
        vocab_list = sorted(vocab_list, key=lambda x: x[1])
        return [v[0] for v in vocab_list]

class PoetryData(Dataset):
    def __init__(self, raw_poetry):
        self.lang = Lang(raw_poetry)
        
        self.data = []
        for line in raw_poetry:
            self.data.append(
                [self.lang.vocab["<sos>"]]+self.lang.str2idx(line) + [self.lang.vocab["<eos>"]]
            )
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index][:-1]
        y = self.data[index][1:]
        return torch.LongTensor(x), torch.LongTensor(y)