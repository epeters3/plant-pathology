import os
from datetime import datetime

import pandas as pd

from pathologist.trainers.transfer_train import transfer_train


def ensemble(train_configs: list) -> None:
    """
    Trains a model for every config in `train_configs`. Takes the predictions
    from each one, and averages them all together.
    """
    # Train each model and get a submission for it.
    submissions = [
        transfer_train(**config, make_submission=True) for config in train_configs
    ]
    # Ensemble the models by averaging their submissions.
    all_submissions = pd.concat(submissions)
    averaged = all_submissions.groupby(all_submissions.index).mean()
    # Add the image_id column back in.
    averaged_submission = pd.concat((submissions[0].image_id, averaged), axis=1)
    # Write out the final ensembled submission.
    write_path = os.path.join("submissions", f"{datetime.now().isoformat()}.csv")
    averaged_submission.to_csv(write_path, index=False)
    print(f"wrote submission to {write_path}")


if __name__ == "__main__":
    ensemble(
        [
            {
                "train_set": "trainsplit-augmented-5",
                "size": 256,
                "epochs": 73,
                "batch_size": 32,
                "learning_rate": 0.00005953,
                "lr_decay_rate": 0.8936,
                "lr_decay_steps": 5705.962,
                "architecture": "BiT-M R101x3",
                "nhiddenlayers": 3,
                "nhiddenunits": 102,
                "l2_regularization": 1.219e-8,
                "dropout_rate": 0.3104,
            },
            {
                "train_set": "trainsplit",
                "size": 256,
                "epochs": 59,
                "batch_size": 32,
                "learning_rate": 0.00000989,
                "lr_decay_rate": 0.8491,
                "lr_decay_steps": 7784.704,
                "architecture": "BiT-M R101x3",
                "nhiddenlayers": 1,
                "nhiddenunits": 249,
                "l2_regularization": 1.436e-8,
                "dropout_rate": 0.1232,
            },
            {
                "train_set": "trainsplit-augmented-5",
                "size": 256,
                "epochs": 68,
                "batch_size": 32,
                "learning_rate": 0.00002363,
                "lr_decay_rate": 0.9182,
                "lr_decay_steps": 7058.96,
                "architecture": "BiT-M R101x3",
                "nhiddenlayers": 3,
                "nhiddenunits": 64,
                "l2_regularization": 0.008376,
                "dropout_rate": 0.5271,
            },
        ]
    )
