import torch
import os
import multiprocessing
import mlflow
import json
from lightning.pytorch.loggers import MLFlowLogger
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.strategies import FSDPStrategy, DeepSpeedStrategy
from torch.distributed.fsdp import StateDictType

import training.metadata.tuning as tuning_metadata
from spam_checker.data.spam_lit_datamodule import SMSDataModule
from spam_checker.models.spam_classifier import SpamClassifier

@rank_zero_only
def log_artifacts_manual(mlflow_client, run_id, folder_path, trainer):
    data = trainer.datamodule
    vocab_pt_file_path = folder_path + "vocab.pt"
    sms_spam_checkpoint_file_path = folder_path + "sms_spam.ckpt"

    if (not hasattr(trainer.strategy, "strategy_name")):
        print("strategy name unknown, most likely ddp")
    else:
        print(trainer.strategy.strategy_name)

    torch.save(data.vocab, vocab_pt_file_path)
    trainer.save_checkpoint(sms_spam_checkpoint_file_path)


    mlflow_client.log_artifacts(
        run_id=run_id, 
        local_dir=folder_path, 
        artifact_path="evaluation"
    )

    print(mlflow.get_run(run_id).info.artifact_uri)

class LogArtifactsCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        # Ensure the logger is an MLFlowLogger

        print("calling on train end")

        if isinstance(trainer.logger, MLFlowLogger):
            # Access the underlying mlflow client and current run ID
            mlflow_client = trainer.logger.experiment
            run_id = trainer.logger.run_id

            data = trainer.datamodule

            folder_path = "./training_run/" + run_id + "/" 

            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

            if os.path.isdir(folder_path):

                print("calling on save artifacts")

                log_artifacts_manual(
                    mlflow_client=mlflow_client,
                    run_id=run_id,
                    folder_path=folder_path,
                    trainer=trainer
                )   

# Example 1: Stop when validation loss stops decreasing
early_stop_loss = EarlyStopping(
    monitor="val_loss",
    patience=3,
    mode="min"
)

# Example 2: Stop when validation accuracy stops increasing
early_stop_acc = EarlyStopping(
    monitor="val_acc", 
    patience=5, 
    min_delta=0.01, # Minimum change to qualify as an improvement
    mode="max"
)

def build_logger(args):

    if not args.mlflow_tracking_uri:
        return None

    # processing = "single"

    # if (args.distributed_processing):
    #     processing = "distributed"

    processing = "distributed"

    initial_hyperparameters = json.dumps({
        "batch_size": args.batch_size,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "max_vocab_size": args.max_vocab_size,
        "max_length": args.max_length,
    })

    mlflow_logger = MLFlowLogger(
        experiment_name="spam_training",
        tracking_uri=args.mlflow_tracking_uri,
        log_model=False,
        tags={
            "processing": processing,
            "initial_hyperparameters": initial_hyperparameters
        }
    )

    return mlflow_logger

def build_callbacks(logger):

    callbacks = [early_stop_loss]

    if logger:
        callbacks.append(LogArtifactsCallback())

    return callbacks

def build_datamodule(args, dataframe):

    dm = SMSDataModule(
        dataframe=dataframe,
        batch_size=args.batch_size,
        num_workers=0,
        max_length=args.max_length,
        max_vocab_size=args.max_vocab_size
    )

    dm.setup()

    return dm

def build_model(args, vocab_size):

    return SpamClassifier(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr
    )

def build_trainer(
        args,
        logger,
        callbacks
):

    # strategy = "auto"

    # if args.distributed_processing:
    #     strategy = "deepspeed"

    strategy = "deepspeed"

    return Trainer(
        max_epochs=args.max_epoch,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=callbacks,
        strategy=strategy,
        num_nodes=args.num_nodes,
        accumulate_grad_batches=4
    )

def train_model(args, dataframe):

    logger = build_logger(args)

    callbacks = build_callbacks(logger)

    dm = build_datamodule(
        args=args,
        dataframe=dataframe
    )

    model = build_model(
        args=args,
        vocab_size=len(dm.vocab)
    )

    trainer = build_trainer(
        args=args,
        logger=logger,
        callbacks=callbacks
    )

    trainer.fit(
        model,
        datamodule=dm
    )

    trainer.test(
        model,
        datamodule=dm
    )