import math

import tensorflow as tf
import wandb
import numpy as np
from tqdm import tqdm
import tensorflow.keras.applications as models

transfer_model_map = {
    "DenseNet121": models.DenseNet121,
    "InceptionV3": models.InceptionV3,
    "ResNet101V2": models.ResNet101V2,
}


class PathologistFineTuneModel(tf.keras.Model):
    """
    An image classification model that allows fine tuning on transfer models.
    The `fit` method trains the entire network, including the
    transfer model. There is also a `fit_head` method which
    trains just the new layers on top of the headless transfer model,
    and includes optimizations.
    """

    def __init__(
        self,
        nclasses: int,
        size: int,
        *,
        architecture: str = "DenseNet121",
        nfinetunelayers: int = 1,
        nhiddenlayers: int = 1,
        nhiddenunits: int = 64,
        l2_regularization: float = 0.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.nfinetunelayers = nfinetunelayers
        self.transfer_model = transfer_model_map[architecture](
            include_top=False, input_shape=(size, size, 3)
        )
        self.transfer_model.trainable = False
        self._set_fine_tune(True)

        self.nn_layers = [self.transfer_model, tf.keras.layers.Flatten()]

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
                "architecture": architecture,
                # "nfinetunelayers": nfinetunelayers,
                "nhiddenlayers": nhiddenlayers,
                "nhiddenunits": nhiddenunits,
                "l2_regularization": l2_regularization,
                "dropout_rate": dropout_rate,
            }
        )

    def _set_fine_tune(self, val: bool) -> None:
        # Allow for fine tuning the last `nfinetunelayers` layers of the transfer
        # learning model.
        for layer in self.transfer_model.layers[-self.nfinetunelayers :]:  # noqa
            layer.trainable = val
        self._fine_tune = val

    def _propagate_transfer(self, X: np.ndarray, batch_size: int = 32) -> tf.Tensor:
        """
        Propagate `X` through just the transfer model.
        """
        nbatches = int(math.ceil(len(X) / batch_size))
        embeddings = []
        batches = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
        for batch in tqdm(batches, total=nbatches):
            embeddings.append(self.transfer_model(batch))
        return tf.concat(embeddings, axis=0)

    def fit_head(self, X, y, *args, validation_data: tuple, **kwargs) -> None:
        """
        Wraps `fit` so that forward propagation through the transfer
        model is only done once for the whole life of the `fit` process.
        This gives a huge speed up and is ok because we are not fine tuning
        the transfer model in this method. Only supports `X`, `y`, and
        `validation_data` as data, and not a Keras flow.
        `validation_data` shoul be a tuple `(X, y)`.
        """
        self._set_fine_tune(False)

        dev_X, dev_y = validation_data
        X_embedding = self._propagate_transfer(X)
        dev_X_embedding = self._propagate_transfer(dev_X)

        # Fit the model using the embedded data.
        self.fit(
            X_embedding, y, *args, validation_data=(dev_X_embedding, dev_y), **kwargs,
        )

        self._set_fine_tune(True)

    def call(self, X):
        """
        Assumes `X` is an embedding when not fine tuning.
        """
        for i, layer in enumerate(self.nn_layers):
            # We skip propagating through the transfer model
            # when we're not fine tuning.
            if i > 0 or self._fine_tune:
                X = layer(X)
        return X
