# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/inference.py

import argparse
import os
import collections

import joblib
import numpy as np
import torch
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_data, get_model

parser = argparse.ArgumentParser()
parser.add_argument("--n_queries", default=2, type=int)
parser.add_argument("--model", default="resnet50", type=str)
parser.add_argument("--data", default="waterbirds", type=str)
parser.add_argument("--savedir", default="exp", type=str)
parser.add_argument("--dfr", action="store_true")
parser.add_argument("--dp", action="store_true")
args = parser.parse_args()


@torch.no_grad()
def run():
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

    # Dataset
    train_ds, _, _, _, n_classes = get_data(args.data)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=4)

    # Infer the logits with multiple queries
    savedir = os.path.join(args.savedir, args.data)
    for path in os.listdir(savedir):
        if os.path.exists(os.path.join(savedir, path, "logits.npy")):
            continue

        print(os.path.join(savedir, path))
        m = get_model(args.model, n_classes)
        w = torch.load(os.path.join(savedir, path, "model.pt"))

        if args.dp:
            m = ModuleValidator.fix(m)
            ModuleValidator.validate(m, strict=False)
            k = w.keys()
            v = w.values()
            k = [key.replace("_module.", "") for key in k]
            w = collections.OrderedDict(zip(k, v))

        m.load_state_dict(w)
        m.to(DEVICE)
        m.eval()

        if args.dfr:
            if args.model == "bert":
                m.classifier = torch.nn.Identity()
            else:
                m.fc = torch.nn.Identity()
            dfr = joblib.load(os.path.join(savedir, path, "dfr.pkl"))

        logits_n = []
        for _ in range(args.n_queries):
            logits = []

            for x, _, _ in tqdm(train_dl):
                x = x.to(DEVICE)

                if args.model == "bert":
                    outputs = m(
                        input_ids=x[:, :, 0],
                        attention_mask=x[:, :, 1],
                        token_type_ids=x[:, :, 2],
                    )[0]  # [0] returns embeddings
                else:
                    outputs = m(x)

                outputs = outputs.cpu().numpy()
                if args.dfr:
                    outputs = dfr.predict_proba(outputs)
                logits.append(outputs)

            logits_n.append(np.concatenate(logits))
        logits_n = np.stack(logits_n, axis=1)
        print(logits_n.shape)  # [n_examples, n_augs, n_classes]

        np.save(os.path.join(savedir, path, "logits.npy"), logits_n)


if __name__ == "__main__":
    run()
