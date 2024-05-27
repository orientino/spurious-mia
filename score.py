import argparse
import multiprocessing as mp
import os

import numpy as np

from utils import get_data

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="waterbirds", type=str)
parser.add_argument("--savedir", default="exp", type=str)
parser.add_argument("--dfr", action="store_true")
args = parser.parse_args()


def load_one(path):
    """
    This function converts the output of `inference.py` using the logit function.
    """
    opredictions = np.load(os.path.join(path, "logits.npy"))  # [n_examples, n_augs, n_classes]

    if args.dfr:
        predictions = opredictions  # DFR output is already a probability vector
    else:
        predictions = opredictions - np.max(opredictions, axis=-1, keepdims=True)
        predictions = np.array(np.exp(predictions), dtype=np.float64)
        predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)

    # Logit function. Be exceptionally careful.
    # Numerically stable everything, as described in the paper.
    train_ds, _, _, _, _ = get_data(args.data)
    labels = train_ds.y_array.numpy()  # (n_examples,)

    COUNT = predictions.shape[0]
    y_true = predictions[np.arange(COUNT), :, labels[:COUNT]]

    print("mean acc", np.mean(predictions[:, 0, :].argmax(1) == labels[:COUNT]))

    predictions[np.arange(COUNT), :, labels[:COUNT]] = 0
    y_wrong = np.sum(predictions, axis=-1)

    logit = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
    np.save(os.path.join(path, "scores.npy"), logit)


def load_stats():
    savedir = os.path.join(args.savedir, args.data)
    with mp.Pool(8) as p:
        p.map(load_one, [os.path.join(savedir, x) for x in os.listdir(savedir)])


if __name__ == "__main__":
    load_stats()
