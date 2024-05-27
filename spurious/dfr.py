import os

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


@torch.no_grad()
def save_embeddings(model, dl, dataname, datasplit, model_dir, device="cpu"):
    """Save embeddings, labels, and groups for a given dataset."""

    if dataname in ["multinli"]:
        model.classifier = torch.nn.Identity()
    else:
        model.fc = torch.nn.Identity()
    model.eval()

    all_embeddings = []
    all_ys = []
    all_gs = []
    for x, y, g in tqdm(dl):
        x = x.to(device)

        if dataname in ["multinli"]:
            embeddings = model(
                input_ids=x[:, :, 0],
                attention_mask=x[:, :, 1],
                token_type_ids=x[:, :, 2],
            )[0]  # [0] returns embeddings
        else:
            embeddings = model(x)
            g = g[:, 0]

        all_embeddings.append(embeddings.cpu().detach().numpy())
        all_ys.append(y.numpy())
        all_gs.append(g.numpy())
    all_embeddings = np.vstack(all_embeddings)
    all_ys = np.concatenate(all_ys)
    all_gs = np.concatenate(all_gs)

    if dataname in ["waterbirds", "celeba"]:
        all_gs = 2 * all_ys + all_gs

    np.savez(
        os.path.join(model_dir, f"embeddings_{datasplit}.npz"),
        embeddings=all_embeddings,
        labels=all_ys,
        groups=all_gs,
    )


def train_dfr(model_dir, dataname, datasplit="train", norm="l1", n_heads=10, samples=200):

    def group_balance_data(x, y, g):
        g_idx = {g_: np.where(g == g_)[0] for g_ in np.unique(g)}
        for g_ in g_idx.values():
            np.random.shuffle(g_)

        if dataname in ["fmow", "multinli"]:
            if dataname == "fmow":
                del g_idx[5]  # remove the `others` group for evaluation
            min_g = np.min([np.min([len(g) for g in g_idx.values()]), samples])
        else:
            min_g = np.min([len(g) for g in g_idx.values()])

        x = np.concatenate([x[g_[:min_g]] for g_ in g_idx.values()])
        y = np.concatenate([y[g_[:min_g]] for g_ in g_idx.values()])
        g = np.concatenate([g[g_[:min_g]] for g_ in g_idx.values()])
        return x, y, g

    group_data_fn = group_balance_data

    # Prepare dataset
    # Sample a group balanced (or unbalanced) subset
    train_set = np.load(os.path.join(model_dir, "embeddings_train.npz"))
    val_set = np.load(os.path.join(model_dir, "embeddings_val.npz"))
    x_train, x_val = train_set["embeddings"], val_set["embeddings"]
    y_train, y_val = train_set["labels"], val_set["labels"]
    g_train, g_val = train_set["groups"], val_set["groups"]
    print(f"Train dataset groups: {np.bincount(g_train)}")
    print(f"Valid dataset groups: {np.bincount(g_val)}")

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    if datasplit == "train":
        x, y, g = group_data_fn(x_train, y_train, g_train)
        while len(np.unique(y)) != len(np.unique(y_val)):
            x, y, g = group_data_fn(x_train, y_train, g_train)
    else:
        x, y, g = x_val, y_val, g_val

    groups = np.unique(g)
    if dataname in ["fmow"]:
        groups = np.delete(groups, np.argwhere(groups==5))  # remove the `others` group for evaluation
    print(f"Balanced dataset: {np.bincount(g)}")
    print(f"Groups: {groups}")

    # Train logistic regression
    # Perform hyperparameter search and combine 10 logistic regression models
    if dataname in ["waterbirds", "celeba"]:
        C_OPTIONS = [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
        CLASS_WEIGHTS = [2.0, 3.0, 10.0, 100.0, 300.0, 1000.0]
        CLASS_WEIGHT_OPTIONS = [{0: 1, 1:1}] + [
            {0: 1, 1: w} for w in CLASS_WEIGHTS] + [
            {0: w, 1: 1} for w in CLASS_WEIGHTS]
    elif dataname == "multinli":
        C_OPTIONS = [1.0, 0.5, 0.1, 0.05, 0.01]
        CLASS_WEIGHTS = [2.0, 3.0, 10.0, 100.0]
        CLASS_WEIGHT_OPTIONS = [{0: 1, 1: 1, 2: 1}] + [
            {0: w, 1: 1, 2: 1} for w in CLASS_WEIGHTS] + [
            {0: 1, 1: w, 2: 1} for w in CLASS_WEIGHTS] + [
            {0: 1, 1: 1, 2: w} for w in CLASS_WEIGHTS]
    elif dataname == "fmow":
        C_OPTIONS = [0.5, 0.1, 0.05]
        CLASS_WEIGHTS = [3.0]
        CLASS_WEIGHT_OPTIONS = [{c: 1 for c in range(62)}]
        for i in range(62):
            for w in CLASS_WEIGHTS:
                _weight = {c: 1 for c in range(62)}
                _weight[i] = w
                CLASS_WEIGHT_OPTIONS += [_weight]
    else:
        raise NotImplementedError
        
    worst_accs = {}
    for c in tqdm(C_OPTIONS):
        for cls_w in CLASS_WEIGHT_OPTIONS:
            logreg = LogisticRegression(norm, C=c, solver="liblinear", max_iter=20, class_weight=cls_w)
            logreg.fit(x, y)
            preds = logreg.predict(x_val)
            group_accs = np.array([(preds == y_val)[g_val == g].mean() for g in groups])
            worst_accs[np.min(group_accs)] = [c, cls_w]

    ks, vs = list(worst_accs.keys()), list(worst_accs.values())
    best_hp = vs[np.argmax(ks)]
    best_hp = {"C": best_hp[0], "class_weight": best_hp[1]}
    print(f"Best hyperparameters: {best_hp} (worst acc: {np.max(ks):.3f})")

    coefs, intercepts = [], []
    for _ in tqdm(range(n_heads)):
        x, y, g = group_data_fn(x_train, y_train, g_train)  
        while len(np.unique(y)) != len(np.unique(y_val)):
            x, y, g = group_data_fn(x_train, y_train, g_train)  

        logreg = LogisticRegression(norm, solver="liblinear", max_iter=20, **best_hp)
        logreg.fit(x, y)
        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

        preds = logreg.predict(x_val)
        group_accs = np.array([(preds == y_val)[g_val == g].mean() for g in groups])
        print(f"Validation WGA:\t {np.min(group_accs)}")
        print(f"Trainset sampled group:\t {np.bincount(g)}")
        print(f"Trainset sampled labels:\t {y[:20]}")

    # Save model
    logreg = LogisticRegression(**best_hp)
    logreg.fit(x_val, y_val)  # dummy call, sklearn must run `fit()` for initialization
    logreg.coef_ = np.mean(coefs, axis=0)
    logreg.intercept_ = np.mean(intercepts, axis=0)
    dfr = Pipeline([("scaler", scaler), ("logreg", logreg)])
    joblib.dump(dfr, os.path.join(model_dir, "dfr.pkl"))

    return dfr


def test_dfr(model_dir, dataname, datasplit):
    dfr = joblib.load(os.path.join(model_dir, "dfr.pkl"))
    ds = np.load(os.path.join(model_dir, f"embeddings_{datasplit}.npz"))
    x, y, g = ds["embeddings"], ds["labels"], ds["groups"]

    groups = np.unique(g)
    if dataname in ["fmow"]:
        groups = np.delete(groups, np.argwhere(groups==5))  # remove the `others` group for evaluation
    print(groups)

    preds = dfr.predict(x)
    group_accs = np.array([(preds == y)[g == g_].mean() for g_ in groups])

    return np.mean(group_accs), np.min(group_accs)
