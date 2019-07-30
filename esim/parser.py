import argparse
import time

def parse_arguments():
    """Parse commom arguments
    Requires: argparse, time
    
    Returns:
        dict -- key-value pair of arguments and its value
    """
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument("train", help="")
    parser.add_argument("valid", help="")

    parser.add_argument("--d_embed", "-de", type=int, default=300, help="")
    parser.add_argument("--d_hidden","-dh", type=int, default=300, help="")
    parser.add_argument("--n_layers","-l", type=int, default=2,help="")
    parser.add_argument("--d_proj", type=int, default=300,help="")
    parser.add_argument("--d_v", type=int, default=300, help="")
    parser.add_argument("--n_layers_cmp", type=int, default=2, help="")
    parser.add_argument("--d_pred", type=int, default=300, help="")

    parser.add_argument("--dropout",type=float, default=0, help="")

    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="")
    parser.add_argument("--epochs", type=int, default=50, help="")
    parser.add_argument("--report_interval","-ri", type=int, default=100,help="")
    parser.add_argument("--cuda",action="store_true", help="")

    parser.add_argument("--tag", help="")
    args = parser.parse_args()

    # post-process of some options
    if not args.tag:
        time_stamp = time.strftime("%y%m%d%H%M%S", time.localtime())  
        args.tag = "exp" + time_stamp

    return args