import torch
import os
import multiprocessing
import mlflow
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

def train_model(
    args,
    df,
    best = tuning_metadata.BASE_PARAMETERS,
    tuned=False
):
    multiprocessing.set_start_method("spawn", force=True)
    mlflow_logger = None

    callbacks = [early_stop_loss]

    if args.mlflow_tracking_uri:
        if (len(args.mlflow_tracking_uri)):
            # mlflow.pytorch.autolog()
            # mlflow.set_tracking_uri(args.mlflow_tracking_uri)

            # processing = "single"

            # if (args.distributed_processing):
            #     processing = "distributed"

            processing = "distributed"

            mlflow_logger = MLFlowLogger(
                experiment_name="spam_training",
                tracking_uri=args.mlflow_tracking_uri,  # Point to your local or remote server
                # log_model='all',
                log_model='False',
                tags={"processing": processing}
            )
            callbacks.append(LogArtifactsCallback())

    # print(callbacks)

    # {'batch_size': 16, 'embedding_dim': 32, 'hidden_dim': 256, 'lr': 0.000585829304703927, 'max_vocab_size': 5000, 'max_length': 57}


    if (not df.empty):

        # print(df)

        # num_workers = args.num_workers

        num_workers = 0

        batch_size = args.batch_size
        max_length = args.max_length
        max_vocab_size = args.max_vocab_size
        embedding_dim = args.embedding_dim
        hidden_dim = args.hidden_dim
        hidden_dim = args.hidden_dim
        lr = args.lr

        if tuned:
            batch_size = best["batch_size"]
            max_length = best["max_length"]
            max_vocab_size = best["max_vocab_size"]
            embedding_dim = best["embedding_dim"]
            hidden_dim = best["hidden_dim"]
            lr = best["lr"]


        max_epochs = args.max_epochs
        accelerator = args.accelerator
        devices = args.devices
        # strategy_args = args.strategy
        strategy = "auto"
        num_nodes = args.num_nodes
        devices = args.devices

        data = SMSDataModule(
            dataframe=df,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=max_length,
            max_vocab_size=max_vocab_size
        )


        data.setup()

        print("<------------------------data load complete-------------------------------->")

        model = SpamClassifier(
            vocab_size=len(data.vocab),
            # **best
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            lr=lr
        )

        print("<------------------------model load complete-------------------------------->")

        gpus = int(torch.cuda.is_available())

        # if (args.distributed_processing):

        #     print("calling distributed processing")
        #     strategy = "deepspeed"

        strategy = "deepspeed"

        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices, 
            logger=mlflow_logger,
            callbacks=callbacks,
            strategy=strategy,
            num_nodes=num_nodes,
            accumulate_grad_batches=4,
        )
        print("<------------------------trainer build complete-------------------------------->")

        trainer.fit(
            model,
            datamodule=data,
        )

        print("<------------------------trainer fit complete-------------------------------->")

        trainer.test(
            model,
            datamodule=data,
        )

        print("<------------------------trainer test complete-------------------------------->")

        has_log_callback = any(isinstance(item, LogArtifactsCallback) for item in callbacks)

        if (not has_log_callback):
            torch.save(data.vocab, "vocab.pt")
            trainer.save_checkpoint("sms_spam.ckpt")
