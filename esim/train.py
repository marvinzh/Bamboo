import torch
from model import ESIM
from torch.utils.data import DataLoader
from data import SNLIData

from parser import parse_arguments
from collections import Counter
import json
import argparse


def read_SNLI(path):
    from nltk.tokenize import word_tokenize

    with open(path) as f:
        data_raw = f.readlines()

    rnts = []
    for d in data_raw:
        data = json.loads(d)
        label = data["gold_label"]
        sent1 = word_tokenize(data["sentence1"])
        sent2 = word_tokenize(data["sentence2"])
        if label in ['contradiction', 'entailment', 'neutral']:
            rnts.append([sent1, sent2, label])

    return rnts


def padding_for_SNLI(batch):
    sent1, sent2, target = zip(*batch)
    sent1 = torch.nn.utils.rnn.pad_sequence(sent1)
    sent2 = torch.nn.utils.rnn.pad_sequence(sent2)
    return sent1, sent2, torch.LongTensor(target)


def train_step(model, xs, ys, loss_fn, optimizer, clip=.3):
    model.train()
    optimizer.zero_grad()

    logits = model(*xs)
    loss = loss_fn(logits, ys)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()
    return loss


if __name__ == "__main__":

    args = parse_arguments()
    snli_train = read_SNLI(args.train)
    snli_valid = read_SNLI(args.valid)

    train_raw = SNLIData(snli_train)
    valid_raw = SNLIData(snli_valid)

    train_data = DataLoader(train_raw, batch_size=args.batch_size, shuffle=True, collate_fn=padding_for_SNLI)
    valid_data = DataLoader(valid_raw)

    model = ESIM(train_raw.lang.size, args.d_embed, args.d_hidden, args.n_layers,
                 args.d_proj, args.d_v, args.n_layers_cmp, args.d_pred, args.dropout)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    if args.cuda:
        model.cuda()

    for epoch in range(args.epochs):
        acc_loss = 0.
        for i, (sent1, sent2, y) in enumerate(train_data):
            if args.cuda:
                sent1 = sent1.cuda()
                sent2 = sent2.cuda()

            loss = train_step(model, [sent1, sent2], y, loss_fn, optim)
            acc_loss += loss
            if i % args.report_interval == 0:
                loss_val = acc_loss / args.report_interval
                acc_loss = 0.
                print("Epoch [%2d/%2d], CE: %.4f" % (epoch, args.epochs, loss_val))
