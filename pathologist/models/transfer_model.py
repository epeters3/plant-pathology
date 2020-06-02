import math
import os

import tensorflow as tf
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow_hub as hub

from pathologist.dataset import PathologyDataset
from pathologist.utils import to_h5, from_h5


class TransferModel(tf.keras.Model):
    """
    An image classification model to use on top of transfer model.
    Includes a `fit_head` method which trains just the new layers on
    top of the headless transfer model, and includes optimizations.
    """

    tf_hub_urls = {
        "BiT-M R101x1": "https://tfhub.dev/google/bit/m-r101x1/1",
        "BiT-M R101x3": "https://tfhub.dev/google/bit/m-r101x3/1",
        "BiT-M R152x4": "https://tfhub.dev/google/bit/m-r152x4/1",
        "EfficientNetB2": "https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1",  # noqa: E501
        "InceptionV3": "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",  # noqa: E501
    }

    cache_root = "transfer-cache"

    def __init__(
        self,
        nclasses: int,
        size: int,
        batch_size: int,
        *,
        architecture: str = "BiT-M R101x1",
        fine_tune: bool = False,
        nhiddenlayers: int = 1,
        nhiddenunits: int = 64,
        l2_regularization: float = 0.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.architecture = architecture
        self.size = size  # the image size (in pixels)
        self.batch_size = batch_size
        self.base_model = hub.KerasLayer(self.tf_hub_urls[self.architecture])
        self.base_model.trainable = fine_tune
        self._bypass_base_model_during_fit = False

        self.nn_layers = [tf.keras.layers.Flatten()]

        # Add the hidden layers
        for i in range(nhiddenlayers):
            self.nn_layers.append(
                tf.keras.layers.Dense(
                    nhiddenunits,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                )
            )
            self.nn_layers.append(tf.keras.layers.BatchNormalization())
            self.nn_layers.append(tf.keras.layers.Activation("relu"))
            self.nn_layers.append(tf.keras.layers.Dropout(dropout_rate))

        # The final classification layer.
        self.nn_layers.append(
            tf.keras.layers.Dense(
                nclasses,
                kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                activation="softmax",
            )
        )

        wandb.config.update(
            {
                "architecture": self.architecture,
                "nhiddenlayers": nhiddenlayers,
                "nhiddenunits": nhiddenunits,
                "l2_regularization": l2_regularization,
                "dropout_rate": dropout_rate,
            }
        )

    def _propagate_transfer(self, X: np.ndarray) -> tf.Tensor:
        """
        Propagate `X` through just the transfer model.
        """
        nbatches = int(math.ceil(len(X) / self.batch_size))
        embeddings = []
        batches = tf.data.Dataset.from_tensor_slices(X).batch(self.batch_size)
        for batch in tqdm(batches, total=nbatches):
            embeddings.append(self.base_model(batch))
        return tf.concat(embeddings, axis=0)

    def fit_head(self, train_name: str, dev_name: str, *args, **kwargs) -> None:
        """
        Wraps `fit` so that forward propagation through the transfer
        model is only done once for the whole life of the `fit` process.
        This gives a huge speed up and is ok because we are not fine tuning
        the transfer model in this method. The train data and dev data are passed
        in as strings identifying their names, as passed to the `PathologyDataset`
        constructor.
        
        This method will first look for a cached embedding of the datasets, computed
        by this model's base transfer model, under the `train_name` and `dev_name`
        directories in this model's cache.
        """
        train_base_model = self.base_model.trainable
        # Temporarily set this to false: we're only training the head.
        self.base_model.trainable = False
        self._bypass_base_model_during_fit = True

        # wandb barks if we don't do this.
        self.base_model.build((self.batch_size, self.size, self.size, 3))

        train_data = self._load_from_cache(train_name)
        dev_data = self._load_from_cache(dev_name)

        # Fit the model using the embedded data.
        self.fit(
            train_data["X"],
            train_data["y"],
            *args,
            validation_data=(dev_data["X"], dev_data["y"]),
            batch_size=self.batch_size,
            **kwargs,
        )

        self._bypass_base_model_during_fit = False
        # Set the trainable status back to whatever it was.
        self.base_model.trainable = train_base_model

    def _get_cache_path(self, dataset_name: str) -> str:
        return os.path.join(
            self.cache_root, self.architecture, f"{dataset_name}-{self.size}.h5"
        )

    def _cache_exists(self, dataset_name: str) -> bool:
        return os.path.isfile(self._get_cache_path(dataset_name))

    def _cache(self, dataset_name: str, data: dict) -> None:
        to_h5(data, self._get_cache_path(dataset_name))

    def _load_from_cache(self, dataset_name: str, has_y: bool = True) -> dict:
        """
        Loads the embeddings and labels for `dataset_name` from this
        model's cache, computing and caching them if they don't already exist.
        """
        if self._cache_exists(dataset_name):
            print(
                f"loading cached embeddings for {dataset_name} dataset "
                f"on the {self.architecture} architecture..."
            )
            return from_h5(self._get_cache_path(dataset_name))
        else:
            # Load the dataset, compute embeddings on its X, then cache it.
            dataset = PathologyDataset(dataset_name, self.size, has_y)
            X_embedding = self._propagate_transfer(dataset.X).numpy()
            print(
                f"caching embeddings for {dataset_name} dataset "
                f"on the {self.architecture} architecture..."
            )
            data = {"X": X_embedding}
            if dataset.has_y:
                # Cache y as well
                data["y"] = dataset.y
            self._cache(dataset_name, data)
            return data

    def predict_on_test(self) -> pd.DataFrame:
        """
        Computes predictions of the fitted model for the test set,
        adding the optimization of only propagating through a
        transfer model once.
        """
        image_ids = PathologyDataset("test", self.size, has_y=False).image_ids
        X = self._load_from_cache("test", has_y=False)["X"]

        print("predicting on test set...")
        self._bypass_base_model_during_fit = True
        test_preds = self.predict(X, batch_size=self.batch_size)
        self._bypass_base_model_during_fit = False

        return pd.DataFrame(
            {
                "image_id": image_ids,
                "healthy": test_preds[:, 0],
                "multiple_diseases": test_preds[:, 1],
                "rust": test_preds[:, 2],
                "scab": test_preds[:, 3],
            }
        )

    def call(self, X):
        if not self._bypass_base_model_during_fit:
            # We bypass the base model during `self.fit_head` to speed up
            # training.
            X = self.base_model(X)

        for layer in self.nn_layers:
            X = layer(X)

        return X
