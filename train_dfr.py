import argparse
import os
import time
from collections import Counter

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader, Subset
from utils import get_data, get_model

from spurious.dfr import save_embeddings, test_dfr, train_dfr

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet50", type=str)
parser.add_argument("--data", default="waterbirds", type=str)
parser.add_argument("--savedir", default="exp", type=str)

# LiRA
parser.add_argument("--n_shadows", default=16, type=int)
parser.add_argument("--shadow_id", default=1, type=int)
parser.add_argument("--pkeep", default=0.5, type=float)

# DFR configs
parser.add_argument("--dfr", action="store_true")
parser.add_argument("--embeddings", action="store_true")
parser.add_argument("--norm", default="l1", type=str)
parser.add_argument("--n_heads", default=10, type=int)
parser.add_argument("--samples", default=10000, type=int)

parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

# Solves out-of-memory when running `save_embedding` with large dataset
# https://pytorch.org/docs/stable/multiprocessing.html
if args.data in ["celeba", "fmow", "multinli"]:
    torch.multiprocessing.set_sharing_strategy("file_system")


def run():
    seed = np.random.randint(0, 1000000000)
    seed ^= int(time.time())
    pl.seed_everything(seed)

    wandb.init(project=f"lira-{args.data}", mode="disabled" if args.debug else "online")
    wandb.config.update(args)

    # Compute the IN / OUT subset per group:
    # We apply the same mechanism as in `train.py` but for each group separately.
    # First we separate the dataset into spurious groups, then assign the keep indices
    # to each spurious group. In the end we combine all together. This process is
    # needed to ensure that each shadow model is trained with the same proportion
    # of spurious groups.

    train_ds, val_ds, test_ds, group_array, n_classes = get_data(args.data)

    keep_inds = []
    keep_bool = np.full((len(train_ds)), False)
    for g in np.unique(group_array):
        group = (group_array == g).nonzero()[0]
        group_size = len(group)
        if args.n_shadows is not None:
            np.random.seed(0)
            keep = np.random.uniform(0, 1, size=(args.n_shadows, group_size))
            order = keep.argsort(0)
            keep = order < int(args.pkeep * args.n_shadows)
            keep = np.array(keep[args.shadow_id], dtype=bool)
            keep = keep.nonzero()[0]
        else:
            keep = np.random.choice(group_size, size=int(args.pkeep * group_size), replace=False)
            keep.sort()
        keep = group[keep]
        keep_inds.extend(keep)
        keep_bool[keep] = True

    print(f"\nPre: Trainset size: {len(train_ds)}")
    print(f"Pre: Trainset size per group:  {Counter(group_array)}")
    print(f"Pre: group array:  {group_array}")

    bs = 32
    train_ds = Subset(train_ds, keep_inds)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4)

    print(f"\nPost: Trainset size: {len(train_ds)}")
    print(f"Post: Trainset size per group:  {Counter(group_array[keep_bool])}\n")

    # Train model
    model = get_model(args.model, n_classes)
    model = model.to(DEVICE)

    savedir = os.path.join(args.savedir, args.data, str(args.shadow_id))
    model.load_state_dict(torch.load(os.path.join(savedir, "model.pt")), strict=False)

    if args.embeddings:
        save_embeddings(model, train_dl, args.data, "train", savedir, DEVICE)
        save_embeddings(model, val_dl, args.data, "val", savedir, DEVICE)
        save_embeddings(model, test_dl, args.data, "test", savedir, DEVICE)

    train_dfr(
        savedir, 
        args.data, 
        "train", 
        norm=args.norm, 
        n_heads=args.n_heads, 
        samples=args.samples
    )
    train_acc, train_acc_w = test_dfr(savedir, args.data, datasplit="train")
    val_acc, val_acc_w = test_dfr(savedir, args.data, datasplit="val")
    test_acc, test_acc_w = test_dfr(savedir, args.data, datasplit="test")

    print(f"[train] acc: {train_acc:.4f}, acc_worst: {train_acc_w:.4f}")
    print(f"[val] acc: {val_acc:.4f}, acc_worst: {val_acc_w:.4f}")
    print(f"[test] acc: {test_acc:.4f}, acc_worst: {test_acc_w:.4f}")
    wandb.log({"train/acc": train_acc})
    wandb.log({"train/acc_worst": train_acc_w})
    wandb.log({"val/acc": val_acc})
    wandb.log({"val/acc_worst": val_acc_w})
    wandb.log({"test/acc": test_acc})
    wandb.log({"test/acc_worst": test_acc_w})


if __name__ == "__main__":
    run()
