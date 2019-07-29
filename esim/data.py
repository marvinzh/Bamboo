import torch
from torch.utils.data import Dataset
from lang import Lang
3
class SNLIData(Dataset):
    def __init__(self, raw, n_vocab=-1):
        super().__init__()
        premise = list(map(lambda x:x[0], raw))
        hypos = list(map(lambda x:x[1], raw))
        labels = list(map(lambda x:x[2], raw))
        corpus = [item for p in premise for item in p] + [item for h in hypos for item in h]
        cnt = Counter(corpus)
        self.lang = Lang(cnt, n_vocab=n_vocab)

        self.premise = list(map(
            lambda x:self.lang.str2idx(x),
            premise
        ))
        self.hypos = list(map(
            lambda x:self.lang.str2idx(x),
            hypos
        ))
        self.label2idx = list(set(labels))
        self.labels = list(map(
            lambda x: self.label2idx.index(x),
            labels
        ))
        

    def __len__(self):
        assert len(self.premise) == len(self.hypos) == len(self.labels)
        return len(self.premise)
    
    def __getitem__(self, index):
        premise = self.premise[index]
        hypos = self.hypos[index]
        target = self.labels[index]
        return torch.LongTensor(premise), torch.LongTensor(hypos), target