import tensorflow as tf
import wandb


class PathologistModel(tf.keras.Model):
    def __init__(
        self,
        nclasses: int,
        *,
        nhiddenlayers: int = 1,
        nhiddenunits: int = 64,
        l2_regularization: float = 0.0,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.nn_layers = []

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
                "nhiddenlayers": nhiddenlayers,
                "nhiddenunits": nhiddenunits,
                "l2_regularization": l2_regularization,
                "dropout_rate": dropout_rate,
            }
        )

    def call(self, X):
        for layer in self.nn_layers:
            X = layer(X)
        return X
