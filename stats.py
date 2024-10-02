# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import csv
import functools
import os
import shutil
from collections import defaultdict

import numpy as np
import scipy.stats
from sklearn.metrics import auc, roc_curve

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="waterbirds", type=str)
parser.add_argument("--file", default="", type=str)
parser.add_argument("--n_models", default=5, type=int)
parser.add_argument("--sdir", default="exp", type=str)  # shadow models dir
parser.add_argument("--tdir", default="exp", type=str)  # target models dir
args = parser.parse_args()


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, thresholds = roc_curve(x, -score, drop_intermediate=False)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    return fpr, tpr, auc(fpr, tpr), acc


def load_data():
    """
    Load our saved scores and then put them into a big matrix.
    """
    global scores, keep
    scores = []
    keep = []

    savedir = os.path.join("exp_test", args.data)
    for path in os.listdir(savedir):
        if path == ".DS_Store":
            continue
        scores.append(np.load(os.path.join(savedir, path, "scores.npy")))
        keep.append(np.load(os.path.join(savedir, path, "keep.npy")))
    scores = np.array(scores)  # [n_shadows, n_examples, n_augs]
    keep = np.array(keep)  # [n_shadows, n_examples]

    global scores_group, keep_group
    scores_group = defaultdict(list)
    keep_group = defaultdict(list)

    for path in os.listdir(savedir):
        if path == ".DS_Store":
            continue
        g = np.load(os.path.join(savedir, path, "groups.npy"))
        s = np.load(os.path.join(savedir, path, "scores.npy"))
        k = np.load(os.path.join(savedir, path, "keep.npy"))
        for group in np.unique(g):
            scores_group[group].append(s[g == group])
            keep_group[group].append(k[g == group])
    if args.data == "fmow":
        del scores_group[5]
        del keep_group[5]
    for group in scores_group.keys():
        scores_group[group] = np.array(
            scores_group[group]
        )  # [n_shadows, n_examples_group, n_augs]
        keep_group[group] = np.array(keep_group[group])  # [n_shadows, n_examples_group]
        # print(scores_group[group].shape)

    return scores, keep


def generate_ours(
    keep,
    scores,
    check_keep,
    check_scores,
    in_size=100000,
    out_size=100000,
    fix_variance=False,
):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    in_size = min(min(map(len, dat_in)), in_size)
    out_size = min(min(map(len, dat_out)), out_size)

    dat_in = np.array([x[:in_size] for x in dat_in])
    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_in)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
        score = pr_in - pr_out

        prediction.extend(score.mean(1))
        answers.extend(ans)

    return prediction, answers


def generate_ours_offline(
    keep,
    scores,
    check_keep,
    check_scores,
    in_size=100000,
    out_size=100000,
    fix_variance=False,
):
    """
    Fit a single predictive model using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    out_size = min(min(map(len, dat_out)), out_size)

    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_out = np.std(dat_out)
    else:
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        score = scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)

        prediction.extend(score.mean(1))
        answers.extend(ans)
    return prediction, answers


def get_stats(fn, keep, scores, ntest, sweep_fn=sweep):
    """
    Generate the ROC curves by using a single test model and the rest to train.
    """

    if ntest == 0:
        train_keep = keep[1:]
        train_scores = scores[1:]
    elif ntest == len(keep) - 1:
        train_keep = keep[:-1]
        train_scores = scores[:-1]
    else:
        train_keep = np.vstack((keep[:ntest], keep[ntest + 1 :]))
        train_scores = np.vstack((scores[:ntest], scores[ntest + 1 :]))
    test_keep = np.expand_dims(keep[ntest], 0)
    test_scores = np.expand_dims(scores[ntest], 0)

    prediction, answers = fn(train_keep, train_scores, test_keep, test_scores)

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr > 0.001)[0][0]]

    # print("AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f" % (auc, acc, low))

    return (acc * 100, auc * 100, low * 100, fpr, tpr)


def save_results(online):
    accs_t, aucs_t, lows_t = [], [], []
    generate_fn = generate_ours if online else generate_ours_offline

    if os.path.exists("exp_test"):
        shutil.rmtree("exp_test")
        os.makedirs("exp_test")

    accs, aucs, lows = defaultdict(list), defaultdict(list), defaultdict(list)
    for i in range(args.n_models):
        # Prepare the shadow and target
        ignore = shutil.ignore_patterns("logits.npy", "model.pt", "*.npz")
        shutil.copytree(
            f"{args.sdir}/{args.data}",
            f"exp_test/{args.data}",
            ignore=ignore,
            dirs_exist_ok=True,
        )
        shutil.copytree(
            f"{args.tdir}/{args.data}/{i}",
            f"exp_test/{args.data}/{i}",
            ignore=ignore,
            dirs_exist_ok=True,
        )

        # Global results
        load_data()
        acc_, auc_, low_, _, _ = get_stats(
            functools.partial(generate_fn, fix_variance=True), keep, scores, i
        )
        accs_t.append(acc_)
        aucs_t.append(auc_)
        lows_t.append(low_)

        # Per group results
        for group in scores_group.keys():
            acc_, auc_, low_, _, _ = get_stats(
                functools.partial(generate_fn, fix_variance=True),
                keep_group[group],
                scores_group[group],
                i,
            )
            accs[group].append(acc_)
            aucs[group].append(auc_)
            lows[group].append(low_)

    # Write results to CSV file
    data = []
    for k in accs.keys():
        data.append(
            {
                "online": online,
                "group": k,
                "accuracy": f"{np.mean(accs[k]):.2f} +/- {np.std(accs[k]) / np.sqrt(args.n_models):.2f}",
                "auroc": f"{np.mean(aucs[k]):.2f} +/- {np.std(aucs[k]) / np.sqrt(args.n_models):.2f}",
                "tpr": f"{np.mean(lows[k]):.2f} +/- {np.std(lows[k]) / np.sqrt(args.n_models):.2f}",
            }
        )
        print(f"Group {k} tpr {data[-1]['tpr']}")
    data.append(
        {
            "online": online,
            "group": "T",
            "accuracy": f"{np.mean(accs_t):.2f} +/- {np.std(accs_t) / np.sqrt(args.n_models):.2f}",
            "auroc": f"{np.mean(aucs_t):.2f} +/- {np.std(aucs_t) / np.sqrt(args.n_models):.2f}",
            "tpr": f"{np.mean(lows_t):.2f} +/- {np.std(lows_t) / np.sqrt(args.n_models):.2f}",
        }
    )
    print(f"Group T tpr {data[-1]['tpr']}")

    if args.file != "":
        filename = f"results/{args.data}_{args.file}.csv"
        with open(filename, "a", newline="") as csvfile:
            fieldnames = ["online", "group", "accuracy", "auroc", "tpr"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:
                writer.writeheader()
            for row in data:
                writer.writerow(row)
                print(row)


if __name__ == "__main__":
    print("LiRA Online =========== \n")
    save_results(online=True)
    print("LiRA Offline =========== \n")
    save_results(online=False)
