from fire import Fire
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
import wandb
from wandb.keras import WandbCallback

from pathologist.models.model import PathologistModel
from pathologist.dataset import load_embeddings
from pathologist import constants


def train_from_cache(
    *,
    architecture: str = "BiT",
    size: int = 256,
    augmentation: int = 1,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    lr_decay_rate: float = 0.99,
    lr_decay_steps: int = 5e2,
    **model_params,
) -> float:
    """
    Trains the model using embeddings that were previously cached.
    """
    train_data, dev_data = load_embeddings(architecture, size, augmentation)

    # Make and init the wandb run.
    wandb.init(project="Plant Pathology", reinit=True)
    wandb.config.update(
        {
            "architecture": architecture,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lr_decay_rate": lr_decay_rate,
            "lr_decay_steps": lr_decay_steps,
            "augmentation": augmentation,
            "size": size,
        }
    )

    model = PathologistModel(nclasses=constants.NCLASSES, **model_params)
    model.compile(
        optimizer=Adam(
            learning_rate=ExponentialDecay(learning_rate, lr_decay_steps, lr_decay_rate)
        ),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    model.fit(
        train_data["X"],
        train_data["y"],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(dev_data["X"], dev_data["y"]),
        callbacks=[WandbCallback(save_model=False)],
    )

    # Log the scores
    train_loss, train_acc = model.evaluate(train_data["X"], train_data["y"])
    _, dev_acc = model.evaluate(dev_data["X"], dev_data["y"])
    wandb.run.summary.update(
        {"final_train_loss": train_loss, "final_train_acc": train_acc}
    )
    wandb.join()
    return dev_acc


if __name__ == "__main__":
    Fire(train_from_cache)
