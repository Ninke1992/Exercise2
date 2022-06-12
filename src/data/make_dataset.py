from __future__ import annotations

from pathlib import Path

import tensorflow as tf
import torch
from loguru import logger

Tensor = torch.Tensor


def get_eeg(data_dir: Path = Path("../../data/raw")) -> Path:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
    datapath = tf.keras.utils.get_file(
        "eeg", origin=url, untar=False, cache_dir=data_dir
    )
    datapath = Path(datapath)
    logger.info(f"Data is downloaded to {datapath}.")
    return datapath
