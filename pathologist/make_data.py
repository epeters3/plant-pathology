import os
import math
import copy

from fire import Fire
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import random

from pathologist.utils import to_h5, get_data_path
from pathologist.dataset import PathologyDataset
from pathologist.constants import pretrained_map


def make_datasets(size: int) -> None:
    """
    `size` will produce image tensors of dimensionality `(size, size, 3)`.
    Makes the train and dev datasets.
    """
    train_data = PathologyDataset("train", size)
    # Split the train data and save it into trainsplit and dev sets.
    # trainsplit is just a subset of train.
    train_data.split(0.7, "trainsplit", "dev", seed=0)


def make_test_set(size: int) -> None:
    PathologyDataset("test", size, has_y=False)


def random_zoom(img: np.ndarray, max_zoom: float) -> np.ndarray:
    """
    Makes a randomly zoomed in copy of `img`, zoomed in by a ratio
    of the image, anywhere in `[0, max_zoom]`.
    """
    h, w, nc = img.shape
    zoom = 1 - random.uniform(0, max_zoom)
    zoom_h = int(h * zoom)
    zoom_w = int(w * zoom)
    cropped = tf.image.random_crop(img, (zoom_h, zoom_w, nc))
    return tf.image.resize(cropped, (h, w))


def randomly_augment(img: np.ndarray) -> np.ndarray:
    """
    Returns a randomly augmented copy of `img`.
    """
    augmented = np.copy(img)
    augmented = tf.image.random_flip_left_right(augmented)
    augmented = tf.image.random_brightness(augmented, max_delta=0.5)
    augmented = random_zoom(augmented, 0.2)
    return augmented.numpy()


def make_augmented_training_set(n_augmented_per_image: int, size: int) -> None:
    """
    Augments the training dataset, creating `n_augmented_per_image`
    new images per image.
    """
    assert n_augmented_per_image > 0
    train_data = PathologyDataset("trainsplit", size)
    new_X = []
    new_y = []
    print(
        f"Making augmented training set ({n_augmented_per_image} "
        "new versions per image)..."
    )
    for i in tqdm(range(train_data.ninstances)):
        img = train_data.X[i]
        label = train_data.y[i]
        new_X.append(img)
        new_y.append(label)
        for _ in range(n_augmented_per_image):
            new_X.append(randomly_augment(img))
            new_y.append(label)

    augmented_train_data = copy.deepcopy(train_data)
    augmented_train_data.X = np.array(new_X)
    augmented_train_data.y = np.array(new_y)
    augmented_train_data.ninstances = len(augmented_train_data.X)
    augmented_train_data.name = f"trainsplit-augmented-{n_augmented_per_image}"
    augmented_train_data.to_intermediate()


def embed(dataname: str, size: int, architecture: str) -> None:
    """
    Embeds `dataset.X` using a pretrained computer vision model identified
    by `architecture` and saves the embeddings along with the targets.
    """
    assert architecture in pretrained_map[size]
    embedding_path = get_data_path(f"{dataname}-{size}-{architecture}-embeddings.h5")

    if os.path.isfile(embedding_path):
        # The embedding for this model on this dataset has already been
        # computed and cached.
        print(f"{embedding_path} already exists, skipping.")
        return

    dataset = PathologyDataset(dataname, size)

    embedder_url = pretrained_map[size][architecture]
    print(
        f"Embedding {dataname} dataset ({size}) using pretrained {architecture} model..."
    )
    embedder = hub.KerasLayer(embedder_url)
    batch_size = 16
    batches = tf.data.Dataset.from_tensor_slices(dataset.X).batch(batch_size)
    nbatches = int(math.ceil(dataset.ninstances / batch_size))
    embeddings = []
    for batch in tqdm(batches, total=nbatches):
        embeddings.append(embedder(batch))
    embeddings = tf.concat(embeddings, axis=0).numpy()
    to_h5({"X": embeddings, "y": dataset.y}, embedding_path)


def embed_all(size: int, *dataset_names) -> None:
    for dataname in dataset_names:
        for architecture in pretrained_map[size]:
            embed(dataname, size, architecture)


if __name__ == "__main__":
    Fire(
        {
            "datasets": make_datasets,
            "embed": embed_all,
            "augment": make_augmented_training_set,
            "test": make_test_set,
        }
    )
