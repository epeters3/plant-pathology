from fire import Fire
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
import wandb
from wandb.keras import WandbCallback

from pathologist.models.transfer_model import TransferModel
from pathologist import constants


def transfer_train(
    *,
    train_set: str,
    size: int = 256,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    lr_decay_rate: float = 0.99,
    lr_decay_steps: int = 5e2,
    make_submission: bool = False,
    **model_params,
):
    """
    Trains a new head on top of a transfer model. No fine tuning of the transfer
    model is conducted. Transfer model embeddings are computed once at the
    beginning of the training run. If `make_submission==True`, returns this model's
    scores on the test set, along with the corresponding image ids.
    """

    # Make and init the wandb run.
    wandb.init(project="Plant Pathology", reinit=True)
    wandb.config.update(
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lr_decay_rate": lr_decay_rate,
            "lr_decay_steps": lr_decay_steps,
            "size": size,
        }
    )

    model = TransferModel(constants.NCLASSES, size, batch_size, **model_params)
    model.compile(
        optimizer=Adam(
            learning_rate=ExponentialDecay(learning_rate, lr_decay_steps, lr_decay_rate)
        ),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    # Train the model (just the new layers on top of the transfer model)
    model.fit_head(
        train_set, "dev", epochs=epochs, callbacks=[WandbCallback(save_model=False)],
    )

    # Log the scores
    wandb.join()

    if make_submission:
        return model.predict_on_test()


if __name__ == "__main__":
    Fire(transfer_train)
