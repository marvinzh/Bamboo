class Lang:
    def __init__(self, corpus, reserved_tokens=[], n_vocab=-1):
        """Language module, extract vocabulary form given corpus
        
        Arguments:
            corpus {collections.Counter} -- Counter object
        
        Keyword Arguments:
            reserved_tokens {list} -- reserved tokens list, will be added after the built-in tokens (default: {[]})
            n_vocab {int} -- # of max vocabulary based on the frequency, -1 means infinity.  (default: {-1})
        """

        vocab_list = ["<pad>", "<unk>", "<sos>", "<eos>"]+reserved_tokens + self._build_vocab(corpus)
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
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
            self.vocab_list.append(token)

    def _build_vocab(self, corpus):
        if isinstance(corpus, Counter):
            vocab_list = list(map(
                lambda x:x[0],
                corpus.most_common()
            ))
            return vocab_list
        else:
            raise Exception("unrecognizable corpus, %s"%type(corpus))