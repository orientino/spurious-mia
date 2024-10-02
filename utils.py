import os

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from bit import KNOWN_MODELS
from convnextv2 import convnextv2_tiny
from pytorch_transformers import BertConfig, BertForSequenceClassification
from torch.utils.data import Dataset, Subset
from wilds import get_dataset


def get_model(modelname, n_classes):
    if modelname == "resnet50":
        # top-1 81.21
        # https://huggingface.co/timm/resnet50.a1_in1k
        return timm.create_model(
            "resnet50.a1_in1k", pretrained=True, num_classes=n_classes, drop_rate=0.1
        )
    elif modelname == "bit_s":
        # top-1 81.30
        # https://github.com/google-research/big_transfer
        m = KNOWN_MODELS["BiT-S-R50x1"](head_size=1000)
        m.load_from(np.load("bit-s-r50x1.npz"))
        m.head.conv = nn.Conv2d(2048, 62, kernel_size=1, bias=True)
        return m
    elif modelname == "convnext_t":
        # top-1
        # param 82.1M
        # https://huggingface.co/timm/convnext_tiny.fb_in1k
        return timm.create_model(
            "timm/convnext_tiny.fb_in1k",
            pretrained=True,
            num_classes=n_classes,
            drop_rate=0.1,
        )
    elif modelname == "convnextv2_t":
        # top-1 83.0 (finetuned version, we use the self-supervised only)
        # param 28.6M
        # https://huggingface.co/facebook/convnextv2-tiny-1k-224
        # https://github.com/facebookresearch/ConvNeXt-V2/tree/main
        m = convnextv2_tiny(num_classes=1000)
        m.load_state_dict(torch.load("convnextv2_tiny_1k_224_ema.pt")["model"])
        m.head = nn.Linear(m.head.in_features, n_classes)
        return m
    elif modelname == "vit_s":
        # top-1 79
        # https://arxiv.org/pdf/2106.10270.pdf (?)
        # https://huggingface.co/timm/vit_small_patch16_224.augreg_in1k
        return timm.create_model(
            "vit_small_patch16_224.augreg_in1k",
            pretrained=True,
            num_classes=n_classes,
            drop_rate=0.1,
        )
    elif modelname == "deit3_s":
        # top-1 81.37
        # https://huggingface.co/timm/deit3_small_patch16_224.fb_in1k
        return timm.create_model(
            "deit3_small_patch16_224.fb_in1k",
            pretrained=True,
            num_classes=n_classes,
            drop_rate=0.1,
        )
    elif modelname == "swin_t":
        # top-1 81.2
        # param 28.0M
        # https://huggingface.co/timm/swin_tiny_patch4_window7_224.ms_in1k
        return timm.create_model(
            "swin_tiny_patch4_window7_224.ms_in1k",
            pretrained=True,
            num_classes=n_classes,
            drop_rate=0.1,
        )
    elif modelname == "hiera_t":
        # top-1 82.78
        # param 27.9M
        # https://huggingface.co/timm/hiera_tiny_224.mae_in1k_ft_in1k
        return timm.create_model(
            "hiera_tiny_224.mae_in1k_ft_in1k",
            pretrained=True,
            num_classes=n_classes,
            drop_rate=0.1,
        )
    elif modelname == "bert":
        config_class = BertConfig
        model_class = BertForSequenceClassification
        config = config_class.from_pretrained(
            "bert-base-uncased", num_labels=3, finetuning_task="mnli"
        )
        return model_class.from_pretrained(
            "bert-base-uncased", from_tf=False, config=config
        )
    else:
        raise NotImplementedError


def get_data(dataname, augment=True):
    datadir = "/scratch/users/czhang/opt/data"

    from pathlib import Path

    datadir = Path.home() / "opt/data"

    normalize = [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    if dataname == "waterbirds":
        scale = 256.0 / 224.0
        resolutions = (224, 224)
        transform_train = T.Compose(
            [
                T.RandomResizedCrop(
                    resolutions,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=2,
                    antialias=True,
                ),
            ]
            + normalize
        )
        transform_test = T.Compose(
            [
                T.Resize(
                    (int(resolutions[0] * scale), int(resolutions[1] * scale)),
                    antialias=True,
                ),
                T.CenterCrop(resolutions),
            ]
            + normalize
        )
        transform_train = transform_train if augment else transform_test
        data = get_dataset("waterbirds", root_dir=datadir, download=True)
        train_ds = data.get_subset("train", transform=transform_train)
        val_ds = data.get_subset("val", transform=transform_test)
        test_ds = data.get_subset("test", transform=transform_test)
        groups = (
            2 * train_ds.metadata_array[:, 1].numpy()
            + train_ds.metadata_array[:, 0].numpy()
        )
    elif dataname == "celeba":
        orig_w = 178
        orig_h = 218
        orig_min_dim = min(orig_w, orig_h)
        target_resolution = (orig_w, orig_h)
        transform_train = T.Compose(
            [
                T.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(1.0, 1.3333333333333333),
                    interpolation=2,
                ),
                T.RandomHorizontalFlip(),
            ]
            + normalize
        )
        transform_test = T.Compose(
            [
                T.CenterCrop(orig_min_dim),
                T.Resize(target_resolution),
            ]
            + normalize
        )
        transform_train = transform_train if augment else transform_test
        data = get_dataset("celebA", root_dir=datadir, download=True)
        train_ds = data.get_subset("train", transform=transform_train)
        val_ds = data.get_subset("val", transform=transform_test)
        test_ds = data.get_subset("test", transform=transform_test)
        groups = (
            2 * train_ds.metadata_array[:, 1].numpy()
            + train_ds.metadata_array[:, 0].numpy()
        )
    elif dataname == "multinli":
        data = MultiNLIDataset(
            os.path.join(datadir, "multinli_1.0"),
            target_name="gold_label_random",
            confounder_names=["sentence2_has_negation"],
            model_type="bert",
        )
        splits = data.get_splits(["train", "val", "test"])
        train_ds, val_ds, test_ds = splits["train"], splits["val"], splits["test"]
        groups = train_ds.group_array
    elif "fmow" in dataname:
        transform_train = T.Compose([T.RandomHorizontalFlip()] + normalize)
        transform_test = T.Compose(normalize)
        transform_train = transform_train if augment else transform_test
        data = get_dataset("fmow", root_dir=datadir, download=True)
        train_ds = data.get_subset("train", transform=transform_train)
        val_ds = data.get_subset("val", transform=transform_test)
        test_ds = data.get_subset("test", transform=transform_test)
        groups = train_ds.metadata_array[:, 0].numpy()
        if dataname == "fmow16":
            train_ds = ReducedDataset(train_ds, 16)
            val_ds = ReducedDataset(val_ds, 16)
            test_ds = ReducedDataset(test_ds, 16)
        elif dataname == "fmow4":
            train_ds = ReducedDataset(train_ds, 4)
            val_ds = ReducedDataset(val_ds, 4)
            test_ds = ReducedDataset(test_ds, 4)
    else:
        raise NotImplementedError

    return train_ds, val_ds, test_ds, groups, data.n_classes


# MultiNLI Dataset --------------------------------------


class MultiNLIDataset(Dataset):
    """
    MultiNLI dataset.
    label_dict = {
        'contradiction': 0,
        'entailment': 1,
        'neutral': 2
    }
    # Negation words taken from https://arxiv.org/pdf/1803.02324.pdf
    negation_words = ['nobody', 'no', 'never', 'nothing']
    """

    def __init__(
        self,
        root_dir,
        target_name,
        confounder_names,
        augment_data=False,
        model_type=None,
    ):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data

        assert len(confounder_names) == 1
        assert confounder_names[0] == "sentence2_has_negation"
        assert target_name in ["gold_label_preset", "gold_label_random"]
        assert augment_data is False
        assert model_type == "bert"

        self.data_dir = os.path.join(self.root_dir, "data")
        self.glue_dir = os.path.join(self.root_dir, "glue_data", "MNLI")
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Please generate the dataset first."
            )
        if not os.path.exists(self.glue_dir):
            raise ValueError(
                f"{self.glue_dir} does not exist yet. Please generate the dataset first."
            )

        # Read in metadata
        type_of_split = target_name.split("_")[-1]
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, f"metadata_{type_of_split}.csv"), index_col=0
        )

        # Get the y values
        # gold_label is hardcoded
        self.y_array = self.metadata_df["gold_label"].values
        self.n_classes = len(np.unique(self.y_array))

        self.confounder_array = self.metadata_df[confounder_names[0]].values
        self.n_confounders = len(confounder_names)

        # Map to groups
        self.n_groups = len(np.unique(self.confounder_array)) * self.n_classes
        self.group_array = (
            self.y_array * (self.n_groups / self.n_classes) + self.confounder_array
        ).astype("int")

        # Extract splits
        self.split_array = self.metadata_df["split"].values
        self.split_dict = {"train": 0, "val": 1, "test": 2}

        # Load features
        self.features_array = []
        for feature_file in [
            "cached_train_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm",
        ]:
            features = torch.load(os.path.join(self.glue_dir, feature_file))
            self.features_array += features

        self.all_input_ids = torch.tensor(
            [f.input_ids for f in self.features_array], dtype=torch.long
        )
        self.all_input_masks = torch.tensor(
            [f.input_mask for f in self.features_array], dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in self.features_array], dtype=torch.long
        )
        self.all_label_ids = torch.tensor(
            [f.label_id for f in self.features_array], dtype=torch.long
        )

        self.x_array = torch.stack(
            (self.all_input_ids, self.all_input_masks, self.all_segment_ids), dim=2
        )

        assert np.all(np.array(self.all_label_ids) == self.y_array)

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        x = self.x_array[idx, ...]
        return x, y, g

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups / self.n_classes)
        c = group_idx % (self.n_groups // self.n_classes)

        attr_name = self.confounder_names[0]
        group_name = f"{self.target_name} = {int(y)}, {attr_name} = {int(c)}"
        return group_name

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ("train", "val", "test"), split + " is not a valid split"
            mask = self.split_array == self.split_dict[split]
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if train_frac < 1 and split == "train":
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
            subsets[split].group_array = self.group_array[indices]
            subsets[split].y_array = torch.from_numpy(self.y_array[indices])
        return subsets


# FMoW Reduced Dataset --------------------------------------
# Change the number of classes for FMoW dataset to control the complexity


class ReducedDataset(Dataset):
    def __init__(self, data, n_classes):
        self.data = data
        self.n_classes = n_classes
        chunk = len(np.unique(self.data.y_array)) / self.n_classes
        self._y_array = np.array(
            [y // chunk for y in self.data.y_array], dtype=np.int64
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, _, metadata = self.data[idx]
        y = self.y_array[idx]
        return x, y, metadata

    @property
    def metadata_array(self):
        return self.data.metadata_array

    @property
    def y_array(self):
        return self._y_array
