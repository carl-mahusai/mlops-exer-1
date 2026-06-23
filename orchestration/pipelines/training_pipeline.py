from argparse import Namespace

from prefect import flow

from orchestration.tasks.training_tasks import (
    run_training_task
)


@flow
def training_pipeline(

        data: str,

        name_of_label_column: str,
        name_of_message_column: str,

        mlflow_tracking_uri: str = "",

        max_epoch: int = 2,

        accelerator: str = "auto",

        devices: int = 1,

        batch_size: int = 16,

        embedding_dim: int = 64,

        hidden_dim: int = 64,

        lr: float = 1e-3,

        max_vocab_size: int = 5000,

        max_length: int = 50,

        optimize_and_train: bool = False,

        # distributed_processing: bool = False,

        num_nodes: int = 1,

        n_trials: int = 20
):

    args = Namespace(
        data=data,
        name_of_label_column=name_of_label_column,
        name_of_message_column=name_of_message_column,
        mlflow_tracking_uri=mlflow_tracking_uri,
        max_epoch=max_epoch,
        accelerator=accelerator,
        devices=devices,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        max_vocab_size=max_vocab_size,
        max_length=max_length,
        optimize_and_train=optimize_and_train,
        # distributed_processing=distributed_processing,
        num_nodes=num_nodes,
        n_trials=n_trials
    )

    run_training_task(args)