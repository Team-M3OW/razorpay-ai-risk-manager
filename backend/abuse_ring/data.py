"""Loads the published CARE-GNN benchmark graphs (FraudYelp / FraudAmazon).

These ship with DGL and download from https://data.dgl.ai/dataset/ on first use
(no Kaggle account, no auth). Fixed train/val/test masks come with the dataset
and are the same splits every paper on this benchmark (CARE-GNN, PC-GNN,
FRAUDRE, BWGNN, ...) evaluates against.
"""

from dgl.data import FraudAmazonDataset, FraudYelpDataset

DATASETS = {"yelp": FraudYelpDataset, "amazon": FraudAmazonDataset}


def load_dataset(name: str):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {list(DATASETS)}")
    g = DATASETS[name]()[0]
    return g
