from fire import Fire
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
import wandb
from wandb.keras import WandbCallback

from pathologist.models.fine_tune_model import PathologistFineTuneModel
from pathologist.stream_data import make_train_data_generator
from pathologist.dataset import PathologyDataset
from pathologist import constants


def fine_tune_train(
    *,
    size: int = 256,
    epochs: int = 20,
    # nfinetuneepochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    lr_decay_rate: float = 0.99,
    lr_decay_steps: int = 5e2,
    **model_params,
) -> float:
    """
    Uses fine tuning of transfer models and data streaming to enhance dataset
    augmentation.
    """
    train_data = PathologyDataset("trainsplit", size)
    dev_data = PathologyDataset("dev", size)
    # train_gen = make_train_data_generator()

    # Make and init the wandb run.
    wandb.init(project="Plant Pathology", reinit=True)
    wandb.config.update(
        {
            "epochs": epochs,
            # "nfinetuneepochs": nfinetuneepochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lr_decay_rate": lr_decay_rate,
            "lr_decay_steps": lr_decay_steps,
            "size": size,
        }
    )

    model = PathologistFineTuneModel(constants.NCLASSES, size, **model_params)
    model.compile(
        optimizer=Adam(
            learning_rate=ExponentialDecay(learning_rate, lr_decay_steps, lr_decay_rate)
        ),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    # Train the model (just the new layers on top of the transfer model)
    model.fit_head(
        train_data.X,
        train_data.y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(dev_data.X, dev_data.y),
        callbacks=[WandbCallback(save_model=False)],
    )

    # # Now fine tune the whole model, including data augmentation.
    # model.fit(
    #     train_gen.flow(train_data.X, train_data.y, batch_size=batch_size),
    #     epochs=nfinetuneepochs,
    #     steps_per_epoch=int(train_data.ninstances / batch_size),
    #     validation_data=(dev_data.X, dev_data.y),
    #     callbacks=[WandbCallback(save_model=False)],
    # )

    # Log the scores
    _, dev_acc = model.evaluate(dev_data.X, dev_data.y)
    wandb.join()
    return dev_acc


if __name__ == "__main__":
    Fire(fine_tune_train)
