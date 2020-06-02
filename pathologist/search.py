"""
Performing hyperparameter search.
"""
from fire import Fire
import random
from pprint import pprint

from pathologist.trainers.train_from_cache import train_from_cache
from pathologist.trainers.fine_tune_train import fine_tune_train
from pathologist.trainers.transfer_train import transfer_train


def make_discrete_sampler(options):
    return lambda: random.choice(list(options))


def make_sampler(lbound, ubound, method: str = "uniform", base: int = 10):
    """
    If `method=="uniform"`, returns a function that when called samples
    uniformly from the range `[lbound, ubound]`. When `method=="log"`,
    returns a function that samples from a log method with the power
    being drawn uniformly from the range `[lbound, ubound]`. E.g.
    `make_sampler(-4,-1,"log")()` will return a number between
    `1e-4` and `1e-1` drawn unifromly from a base 10 log method. When
    `method=="int"`, returns a function that when called samples an
    integer uniformly from the range `[lbound, ubound]`.
    """
    if method == "uniform":
        return lambda: random.uniform(lbound, ubound)
    elif method == "log":
        return lambda: base ** random.uniform(lbound, ubound)
    elif method == "int":
        return lambda: random.randint(lbound, ubound)
    else:
        raise ValueError("unsupported sample method")


# Hyperparams for `pathology.train_from_cache.train_from_cache`
train_from_cache_hyperparam_samplers = {
    "size": make_discrete_sampler({256}),
    "augmentation": make_discrete_sampler({None, 1, 5}),
    "epochs": make_sampler(10, 200, "int"),
    "learning_rate": make_sampler(-6, -1, "log"),
    "lr_decay_rate": make_sampler(0.80, 1.0),
    "lr_decay_steps": make_sampler(1e1, 1e4),
    "l2_regularization": make_sampler(-10, -1, "log"),
    "dropout_rate": make_sampler(0.0, 0.8),
    "architecture": make_discrete_sampler({"BiT", "ResNetV2-101", "InceptionV3"}),
    "nhiddenlayers": make_sampler(0, 3, "int"),
    "nhiddenunits": make_sampler(16, 256, "int"),
}

# Hyperparams for `pathology.fine_tune_train.fine_tune_train`
fine_tune_train_hyperparam_samplers = {
    "size": make_discrete_sampler({256}),
    "epochs": make_sampler(10, 150, "int"),
    # "nfinetuneepochs": make_sampler(5, 50, "int"),
    # "nfinetunelayers": make_sampler(0, 5, "int"),
    "learning_rate": make_sampler(-6, -1, "log"),
    "lr_decay_rate": make_sampler(0.80, 1.0),
    "lr_decay_steps": make_sampler(1e1, 1e4),
    "l2_regularization": make_sampler(-10, -1, "log"),
    "dropout_rate": make_sampler(0.0, 0.8),
    "architecture": make_discrete_sampler(
        {"DenseNet121", "InceptionV3", "ResNet101V2"}
    ),
    "nhiddenlayers": make_sampler(0, 3, "int"),
    "nhiddenunits": make_sampler(16, 256, "int"),
}

transfer_train_hyperparam_samplers = {
    "train_set": make_discrete_sampler(
        {"trainsplit", "trainsplit-augmented-1", "trainsplit-augmented-5"}
    ),
    "size": make_discrete_sampler({256}),
    "epochs": make_sampler(10, 150, "int"),
    "learning_rate": make_sampler(-6, -1, "log"),
    "lr_decay_rate": make_sampler(0.80, 1.0),
    "lr_decay_steps": make_sampler(1e1, 1e4),
    "l2_regularization": make_sampler(-10, -1, "log"),
    "dropout_rate": make_sampler(0.0, 0.8),
    "architecture": make_discrete_sampler(
        {"BiT-M R101x3", "BiT-M R101x1", "EfficientNetB2", "InceptionV3"}
    ),
    "nhiddenlayers": make_sampler(0, 3, "int"),
    "nhiddenunits": make_sampler(16, 256, "int"),
}


def sample_params(hyperparam_samplers: dict) -> dict:
    return {
        param_name: sampler() for param_name, sampler in hyperparam_samplers.items()
    }


def search(*, nsamples: int, strategy: str) -> dict:
    """
    Randomly samples `nsamples` models and trains them on the
    training data.
    """
    if strategy == "embeddings":
        trainer = train_from_cache
        samplers = train_from_cache_hyperparam_samplers
    elif strategy == "fine-tune":
        trainer = fine_tune_train
        samplers = fine_tune_train_hyperparam_samplers
    elif strategy == "transfer":
        trainer = transfer_train
        samplers = transfer_train_hyperparam_samplers
    else:
        raise ValueError("unsupported training strategy")

    # Sample and train `nsamples` different models. The results
    # and configs are saved to wandb.
    for _ in range(nsamples):
        params = sample_params(samplers)
        print(f"training {strategy} model with params:")
        pprint(params)
        trainer(**params)


if __name__ == "__main__":
    Fire(search)
