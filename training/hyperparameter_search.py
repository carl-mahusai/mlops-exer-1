
import mlflow
import optuna
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

# from mlflow.utils.mlflow_tags import generate_run_name

# from optuna_integration import PyTorchLightningPruningCallback

import time


from spam_checker.data.spam_lit_datamodule import SMSDataModule
from spam_checker.models.spam_classifier import SpamClassifier

import random

adjectives = [
    "charming", "brave", "silent", "quick", "bold",
    "bright", "calm", "proud", "gentle", "lucky"
]

animals = [
    "fox", "wolf", "eagle", "otter", "lion",
    "tiger", "bear", "hawk", "deer", "lynx"
]

def generate_mlflow_like_name():
    return f"{random.choice(adjectives)}-{random.choice(animals)}-{random.randint(10,99)}"

# print(generate_mlflow_like_name())

def objective(trial, dataframe, args, name_prefix):

    print("checking for hyperparameters")

    # Hyperparameters to optimize
    batch_size = trial.suggest_categorical(
        "batch_size",
        [8, 16, 32, 64]
    )

    embedding_dim = trial.suggest_categorical(
        "embedding_dim",
        [32, 64, 128, 256]
    )

    hidden_dim = trial.suggest_categorical(
        "hidden_dim",
        [32, 64, 128, 256]
    )

    lr = trial.suggest_float(
        "lr",
        1e-5,
        1e-2,
        log=True
    )

    max_vocab_size = trial.suggest_categorical(
        "max_vocab_size",
        [5000, 10000, 20000]
    )

    max_length = trial.suggest_int(
        "max_length",
        25,
        100
    )

    # Create DataModule
    dm = SMSDataModule(
        dataframe=dataframe,
        batch_size=batch_size,
        max_vocab_size=max_vocab_size,
        max_length=max_length,
        num_workers=0
    )

    # Build vocab and datasets
    dm.setup()

    # Create model
    model = SpamClassifier(
        vocab_size=len(dm.vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        lr=lr
    )

    accelerator = args.accelerator
    devices = args.devices
    # strategy_args = args.strategy
    strategy = "auto"
    num_nodes = args.num_nodes
    devices = args.devices

    # if (args.distributed_processing):

    #     print("calling distributed processing")
    #     strategy = "deepspeed"

    strategy = "deepspeed"

    # Create one MLflow run per trial
    mlflow_logger = False
    # callbacks = [
    #     PyTorchLightningPruningCallback(
    #         trial,
    #         monitor="val_loss"
    #     )
    # ]
    if args.mlflow_tracking_uri:
        if (len(args.mlflow_tracking_uri)):


            # processing = "single"

            # if (args.distributed_processing):
            #     processing = "distributed"

            processing = "distributed"

            # name_prefix = generate_run_name()
            # name_prefix = generate_mlflow_like_name()

            

            mlflow_logger = MLFlowLogger(
                tracking_uri=args.mlflow_tracking_uri,  # Point to your local or remote server
                tags={"processing": processing},
                experiment_name="spam_classifier_optuna",
                run_name=f"{name_prefix}_trial_{trial.number}",
                log_model=False,
            )

            mlflow_logger.log_hyperparams({
                "trial_number": trial.number,
                "batch_size": batch_size,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "lr": lr,
                "max_vocab_size": max_vocab_size,
                "max_length": max_length,
            })

    trainer = L.Trainer(
        max_epochs=10,
        # logger=False,
        logger=mlflow_logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator=accelerator,
        devices=devices, 
        # callbacks=callbacks,
        strategy=strategy,
        # num_nodes=num_nodes,
        accumulate_grad_batches=4,
    )

    # trainer = L.Trainer(
    #     max_epochs=10,
    #     logger=False,
    #     enable_checkpointing=False,
    #     enable_progress_bar=False,
    #     accelerator="auto",
    #     devices=1
    # )

    trainer.fit(model, dm)

    val_loss = trainer.callback_metrics["val_loss"].item()

    return val_loss

def hyperparameter_search(
        dataframe,
        args
):

    study = optuna.create_study(
        direction="minimize"
    )
    name_prefix = time.time_ns()

    study.optimize(
        lambda trial:
            objective(
                trial=trial,
                dataframe=dataframe,
                args=args,
                name_prefix=name_prefix
            ),
        n_trials=args.n_trials
    )

    print("Best parameters:")
    print(study.best_params)

    # Create a new MLflow run containing the study results
    if args.mlflow_tracking_uri:
        # mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment("spam_classifier_optuna")

        with mlflow.start_run(run_name=f"{name_prefix}_best_hyperparameters"):
            mlflow.log_params(study.best_params)

            # Optional additional information
            mlflow.log_metric("best_value", study.best_value)
            mlflow.log_param("best_trial_number", study.best_trial.number)

    return study.best_params