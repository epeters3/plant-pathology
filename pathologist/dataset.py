import os
import copy

import pandas as pd
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

from pathologist.utils import (
    get_dir_files,
    object_from_h5,
    object_to_h5,
    from_h5,
    get_data_path,
)


def load_embeddings(architecture: str, size: int, augmentation: int = None) -> tuple:
    """
    `train` and `dev` are each dictionaries with
    keys `"X"` and `"y"`.
    """
    print(f"Loading embeddings for {architecture} architecture...")
    if augmentation:
        train = from_h5(
            get_data_path(
                f"trainsplit-augmented-{augmentation}-{size}"
                f"-{architecture}-embeddings.h5"
            )
        )
    else:
        train = from_h5(
            get_data_path(f"trainsplit-{size}-{architecture}-embeddings.h5")
        )
    dev = from_h5(get_data_path(f"dev-{size}-{architecture}-embeddings.h5"))
    return train, dev


class PathologyDataset:
    """
    Class for loading the Kaggle Plant Pathology images as
    a dataset.
    """

    def __init__(self, name: str, size: int, has_y: bool = True) -> None:
        self.name = name
        self.size = size
        self.has_y = has_y
        self.fields_to_save = ["ninstances", "size", "X"]
        if self.has_y:
            self.fields_to_save.append("y")
        else:
            self.fields_to_save.append("image_ids")
        self.classes = ["healthy", "multiple_diseases", "rust", "scab"]

        if os.path.isfile(self.intermediate_path):
            self._from_intermediate()
        else:
            self._from_raw()
            self.to_intermediate()

    def split(self, ratio: float, namea: str, nameb: str, seed: int = None) -> tuple:
        """
        Splits the dataset randomly into two, with the first dataset having
        `ratio`% of the instances and the second having `1-ratio`%.
        """
        assert ratio > 0 and ratio < 1
        if seed:
            np.random.seed(seed)

        indices = np.random.permutation(np.arange(self.ninstances))
        slice_point = int(ratio * self.ninstances)
        a_indices = indices[:slice_point]
        b_indices = indices[slice_point:]

        return self._make_subset(a_indices, namea), self._make_subset(b_indices, nameb)

    def _make_subset(self, indices: np.ndarray, name: str) -> "PathologyDataset":
        """
        Makes a copy of `self`, but only with a subset of the data,
        indexed by `indices`.
        """
        data = copy.deepcopy(self)
        data.X = data.X[indices]
        if data.has_y:
            data.y = data.y[indices]
        data.ninstances = len(data.X)
        data.name = name
        data.to_intermediate()
        return data

    def to_intermediate(self, overwrite: bool = False) -> None:
        object_to_h5(self, self.intermediate_path, self.fields_to_save, overwrite)

    def _from_intermediate(self) -> None:
        print(f"loading {self.name} dataset from {self.intermediate_path}")
        object_from_h5(self, self.intermediate_path)

    def _from_raw(self):
        # Process and image file names and labels so they're aligned.
        labels = pd.read_csv(f"{self.name}.csv").sort_values("image_id")
        if self.has_y:
            self.y = labels[self.classes].values

        # Load, transform, and embed all images for this name.
        X = []
        fnames = sorted(
            fname
            for fname in get_dir_files("images")
            # Only load images for this name.
            if fname.lower().startswith(self.name)
        )
        if not self.has_y:
            self.image_ids = [fname.split(".")[0] for fname in fnames]
        print(f"Processing {self.name} dataset images...")
        for fname in tqdm(fnames):
            img = load_img(
                os.path.join("images", fname), target_size=(self.size, self.size)
            )
            img_arr = img_to_array(img)
            X.append(img_arr)

        self.ninstances = len(X)
        # Turn the instances into a single 2D tensor.
        self.X = np.array(X) / 255.0

    @property
    def intermediate_path(self) -> str:
        return get_data_path(f"{self.name}-{self.size}-data.h5")
