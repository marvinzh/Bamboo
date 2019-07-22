import torch
from data import padding_for_poetry, PoetryData
import argparse
from rnnlm import RNNLM
from torch.utils.data import DataLoader
import visdom
import math
import time

def parse_arguments():
    """Parse commom arguments
    Requires: argparse, time
    
    Returns:
        dict -- key-value pair of arguments and its value
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="")
    parser.add_argument("--d_embed", "-de", type=int, default=128, help="")
    parser.add_argument("--d_hidden","-dh", type=int, default=256, help="")
    parser.add_argument("--n_layers","-l", type=int, default=2,help="")
    parser.add_argument("--dropout",type=float, default=0, help="")

    parser.add_argument("--data", help="")
    parser.add_argument("--epochs", type=int, default=50, help="")
    parser.add_argument("--report_interval","-ri", type=int, default=100,help="")
    parser.add_argument("--cuda",action="store_true", help="")

    parser.add_argument("--tag", help="")
    args = parser.parse_args()

    if not args.tag:
        time_stamp = time.strftime("%y%m%d%H%M%S", time.localtime())  
        args.tag = "exp" + time_stamp

    return args

def train_batch(model, xs, ys, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()
    
    os = model(xs)
    os = os.view(-1, model.n_vocab)
    ys = ys.view([-1])
    
    loss = loss_fn(os, ys)
    loss.backward()
    optimizer.step()
    
    return loss

def read_raw_data(path):
    with open(path) as f:
        data = f.readlines()
    
    poetrys = list(map(
        lambda x:x.strip().replace("[","").replace("]",""),
        data
    ))

    return poetrys

def plot_loss(vis_env, loss_records):
    x = torch.tensor(range(len(loss_records)))
    y = torch.tensor(loss_records)

    vis_env.line(X=x, Y=y, win="loss")

def plot_example(vis_env, examples):
    string = ""
    for i, ex in enumerate(examples):
        string+="Sample (%d epoch):<br>"%i + ex + "<br><br>"

    vis_env.text(string, win="example")

if __name__=="__main__":

    args = parse_arguments()
    vis_env = visdom.Visdom(env=args.tag)

    poetrys = read_raw_data(args.data)
    poetry_data = PoetryData(poetrys)
    train_data = DataLoader(poetry_data,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=lambda x:padding_for_poetry(x, poetry_data.lang.vocab["<pad>"])
                        )

    rnnlm = RNNLM(poetry_data.lang.size, args.d_embed, args.d_hidden, args.n_layers, args.dropout)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(rnnlm.parameters())

    if args.cuda:
        rnnlm.cuda()

    examples=[]
    loss_records=[]
    for epoch in range(args.epochs):
        acc_loss=0.
        for i, (x, y) in enumerate(train_data):
            if args.cuda:
                x=x.cuda()
                y=y.cuda()

            loss=train_batch(rnnlm, x, y, loss_fn, optim)
            loss_records.append(loss)
            acc_loss+=loss

            if i%args.report_interval ==0:
                ppl = math.exp(acc_loss / args.report_interval)
                acc_loss=0.
                print("Epoch [%2d/%2d], Perplexity: %.4f"%(epoch, args.epochs, ppl))
                plot_loss(vis_env,loss_records)

        example = rnnlm.sampling(poetry_data.lang)
        examples.append("".join(example[1:]))
        plot_example(vis_env, examples)
        print("store checkpoint: exp/ckpt.%d"%epoch)
        torch.save(rnnlm.state_dict(), "exp/ckpt.%d"%epoch)
    
