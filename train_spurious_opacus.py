# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py

import argparse
import os
import time
from collections import Counter, defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from spurious.groupdro import LossComputer, get_loader
from utils import get_data, get_model

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--wd", default=0.1, type=float)
parser.add_argument("--bs", default=32, type=int)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--model", default="resnet50", type=str)
parser.add_argument("--data", default="waterbirds", type=str)
parser.add_argument("--sched", action="store_true")

# LiRA attack
parser.add_argument("--n_shadows", default=16, type=int)
parser.add_argument("--shadow_id", default=1, type=int)
parser.add_argument("--pkeep", default=0.5, type=float)
parser.add_argument("--savedir", default="exp", type=str)

# Spurious robust training
parser.add_argument("--dro", action="store_true")
parser.add_argument("--c", type=int, default=0)
parser.add_argument("--dfr", action="store_true")

# Differential privacy
parser.add_argument("--eps", default=0, type=float)
parser.add_argument("--delta", default=0, type=float)
parser.add_argument("--clip", default=0, type=float)
parser.add_argument("--patience", default=10, type=int)

parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")


def run():
    seed = np.random.randint(0, 1000000000)
    seed ^= int(time.time())
    pl.seed_everything(seed)

    wandb.init(project=f"lira-{args.data}", mode="disabled" if args.debug else "online")
    wandb.config.update(args)

    """
    Dataset. Compute the IN / OUT subset per group:
    We apply the same mechanism as in `train.py` but for each group separately.
    First we separate the dataset into spurious groups, then assign the keep indices
    to each spurious group. In the end we combine all together. This process is
    needed to ensure that each shadow model is trained with the same proportion
    of spurious groups.
    """
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
            keep = np.random.choice(
                group_size, size=int(args.pkeep * group_size), replace=False
            )
            keep.sort()
        keep = group[keep]
        keep_inds.extend(keep)
        keep_bool[keep] = True

    print(f"\nPre: Trainset size: {len(train_ds)}")
    print(f"Pre: size per group:  {Counter(group_array)}")
    print(f"Pre: group array:  {group_array}")

    kwargs = {"num_workers": 4, "pin_memory": False}
    train_ds = Subset(train_ds, keep_inds)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.bs, **kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, batch_size=16, **kwargs)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=16, **kwargs)

    print(f"\nPost: Trainset size: {len(train_ds)}")
    print(f"Post: Trainset size per group:  {Counter(group_array[keep_bool])}\n")

    """
    Standard trainining pipeline.
    We include spurious robust and differential privacy methods.
    """

    m = get_model(args.model, n_classes)
    m = m.to(DEVICE)

    dp = args.eps > 0 and args.delta > 0 and args.clip > 0
    if dp:
        m = ModuleValidator.fix(m)
        ModuleValidator.validate(m, strict=False)
    if args.dro:
        train_ds.group_array = group_array[keep_bool]
        train_dl = get_loader(train_ds, train=True, reweight_groups=True, **kwargs)
        loss_dro = LossComputer(
            torch.nn.CrossEntropyLoss(reduction="none"),
            is_robust=True,
            data=train_ds,
            alpha=0.01,
            gamma=0.1,
            adj=args.c,
            step_size=0.01,
        )

    optim = torch.optim.SGD(m.parameters(), args.lr, 0.9, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    if dp:
        privacy_engine = PrivacyEngine(accountant="rdp")
        m, optim, train_dl = privacy_engine.make_private_with_epsilon(
            module=m,
            optimizer=optim,
            data_loader=train_dl,
            epochs=args.epochs,
            target_epsilon=args.eps,
            target_delta=args.delta,
            max_grad_norm=args.clip,
            clipping="flat",
        )
        savename = f"{args.shadow_id}_eps{args.eps}_clip{args.clip}_lr{args.lr}"
    else:
        savename = str(args.shadow_id)
    savedir = os.path.join(args.savedir, args.data, savename)
    os.makedirs(savedir, exist_ok=True)
    np.save(os.path.join(savedir, "keep.npy"), keep_bool)
    np.save(os.path.join(savedir, "groups.npy"), group_array)

    acc_hist = []
    acc_worst_hist = []
    best_acc = -1
    for epoch in range(args.epochs):
        m.train()
        loss_total = 0
        accs_total = 0
        # pbar = tqdm(train_dl)
        with BatchMemoryManager(
            data_loader=train_dl, max_physical_batch_size=32, optimizer=optim
        ) as safe_dl:
            for i, (x, y, g) in enumerate(tqdm(safe_dl)):
                x, y = x.to(DEVICE), y.to(DEVICE)

                pred = m(x)
                if args.dro:
                    g = g[:, 0].to(DEVICE)
                    g = 2 * y + g if args.data in ["waterbirds", "celeba"] else g
                    loss = loss_dro.loss(pred, y, g, True)
                else:
                    loss = F.cross_entropy(pred, y)

                loss_total += loss.item()
                accs_total += (torch.argmax(pred, dim=1) == y).sum().item()
                # pbar.set_postfix_str(f"loss: {loss:.2f}")

                loss.backward()
                # optimizer won't actually make a step unless logical batch is over
                optim.step()
                # optimizer won't actually clear gradients unless logical batch is over
                optim.zero_grad()

        if args.sched:
            sched.step()

        wandb.log({"train/lr": sched.get_last_lr()[0]})
        wandb.log({"train/acc": accs_total / len(train_ds)})
        wandb.log({"train/loss": loss_total / len(train_dl)})

        loss, acc_weight, acc_worst = evaluate_groups(m, val_dl, DEVICE)
        wandb.log({"val/loss": loss})
        wandb.log({"val/acc": acc_weight})
        wandb.log({"val/acc_worst": acc_worst})
        print(f"[val] ep {epoch}, acc {acc_weight:.4f}, acc_worst {acc_worst:.4f}")

        # Save the best model
        acc_hist.append(acc_weight)
        acc_worst_hist.append(acc_worst)
        if acc_worst > best_acc:
            patience = args.patience
            best_acc = acc_worst
            torch.save(m.state_dict(), os.path.join(savedir, "model.pt"))
            wandb.log({"train/best_epoch": epoch})
            continue

        # Stop training if both `acc_weight` and `acc_worst` are not improving anymore
        if patience == 0:
            wandb.log({"train/last_epoch": epoch})
            break
        patience -= 1
        print(f"[val] patience {patience}")

    # Test model and save
    m.load_state_dict(torch.load(os.path.join(savedir, "model.pt")))
    loss, acc_weight, acc_worst = evaluate_groups(m, train_dl, DEVICE)
    wandb.log({"train/loss": loss})
    wandb.log({"train/acc": acc_weight})
    wandb.log({"train/acc_worst": acc_worst})
    print(f"[train]\t acc {acc_weight:.4f}, acc_worst {acc_worst:.4f}")
    loss, acc_weight, acc_worst = evaluate_groups(m, val_dl, DEVICE)
    wandb.log({"val/loss": loss})
    wandb.log({"val/acc": acc_weight})
    wandb.log({"val/acc_worst": acc_worst})
    print(f"[val]\t acc {acc_weight:.4f}, acc_worst {acc_worst:.4f}")
    loss, acc_weight, acc_worst = evaluate_groups(m, test_dl, DEVICE)
    wandb.log({"test/loss": loss})
    wandb.log({"test/acc": acc_weight})
    wandb.log({"test/acc_worst": acc_worst})
    print(f"[test]\t acc {acc_weight:.4f}, acc_worst {acc_worst:.4f}")


@torch.no_grad()
def evaluate_groups(model, dl, device):
    model.eval()
    losses, preds, ys, gs = [], [], [], []

    for x, y, g in dl:
        x, y, g = x.to(device), y.to(device), g.to(device)

        if args.model == "bert":
            pred = model(
                input_ids=x[:, :, 0],
                attention_mask=x[:, :, 1],
                token_type_ids=x[:, :, 2],
                labels=y,
            )[1]  # [1] returns logits
        else:
            pred = model(x)
            g = g[:, 0]  # for wilds, first metadata index is group metadata

        loss = F.cross_entropy(pred, y)
        _, pred = pred.data.max(1)

        losses.append(loss.item())
        preds.append(pred)
        ys.append(y)
        gs.append(g)
    preds = torch.cat(preds)
    ys = torch.cat(ys)
    gs = torch.cat(gs)

    # depending on the dataset, groups are different
    if args.data in ["waterbirds", "celeba"]:
        gs = 2 * ys + gs

    groups = defaultdict(list)
    for g, y, pred in zip(gs, ys, preds):
        groups[g.item()].append(y.item() == pred.item())
    if args.data == "fmow":
        del groups[5]  # for fmow, group 5 `others` is not used in eval

    weighted_acc = 0
    accuracies = []
    for _, group_preds in groups.items():
        accuracy = sum(group_preds) / len(group_preds)
        accuracies.append(accuracy)
        weighted_acc += accuracy * len(group_preds)
    weighted_acc /= len(preds)

    return np.mean(losses), weighted_acc, min(accuracies)


if __name__ == "__main__":
    run()
