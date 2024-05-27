import os

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms.v2 as T
from pytorch_transformers import BertConfig, BertForSequenceClassification
from torch.utils.data import Dataset, Subset
from wilds import get_dataset


def get_model(modelname, n_classes):
    if modelname == "resnet50":
        # top-1 81.214
        # https://huggingface.co/timm/resnet50.a1_in1k
        return timm.create_model(
            "resnet50.a1_in1k", pretrained=True, num_classes=n_classes, drop_rate=0.1
        )
    elif modelname == "convnext_t":
        # top-1 82.066
        # https://huggingface.co/timm/convnext_tiny.fb_in1k
        return timm.create_model(
            "timm/convnext_tiny.fb_in1k",
            pretrained=True,
            num_classes=n_classes,
            drop_rate=0.1,
        )
    elif modelname == "vit_s":
        # top-1 79 (? https://arxiv.org/pdf/2106.10270.pdf)
        # https://huggingface.co/timm/vit_small_patch16_224.augreg_in1k
        return timm.create_model(
            "vit_small_patch16_224.augreg_in1k",
            pretrained=True,
            num_classes=n_classes,
            drop_rate=0.1,
        )
    elif modelname == "deit3_s":
        # top-1 81.370
        # https://huggingface.co/timm/deit3_small_patch16_224.fb_in1k
        return timm.create_model(
            "deit3_small_patch16_224.fb_in1k",
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
    datadir = # TODO: add dataset dir path

    if dataname == "waterbirds":
        scale = 256.0 / 224.0
        resolutions = (224, 224)
        transform_train = T.Compose(
            [
                T.RandomResizedCrop(
                    resolutions, (0.7, 1.0), (0.75, 1.3333), 2, antialias=True
                ),
                T.RandomHorizontalFlip(),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        transform_test = T.Compose(
            [
                T.Resize(
                    (int(resolutions[0] * scale), int(resolutions[1] * scale)),
                    antialias=True,
                ),
                T.CenterCrop(resolutions),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
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
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        transform_test = T.Compose(
            [
                T.CenterCrop(orig_min_dim),
                T.Resize(target_resolution),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
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
    elif dataname == "fmow":
        transform_train = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # default values
            ]
        )
        transform_test = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # default values
            ]
        )
        transform_train = transform_train if augment else transform_test
        data = get_dataset("fmow", root_dir=datadir, download=True)
        train_ds = data.get_subset("train", transform=transform_train)
        val_ds = data.get_subset("val", transform=transform_test)
        test_ds = data.get_subset("test", transform=transform_test)
        groups = train_ds.metadata_array[:, 0].numpy()
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
