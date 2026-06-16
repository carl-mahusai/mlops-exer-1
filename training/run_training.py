import os
import argparse
import torch
import pandas as pd
import multiprocessing
import mlflow
from lightning.pytorch.loggers import MLFlowLogger
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from spam_checker.data.spam_lit_datamodule import SMSDataModule
from spam_checker.models.spam_classifier import SpamClassifier


def convert_to_df(file_path):
    try:
        return pd.read_csv(file_path, encoding_errors='ignore') # or pd.read_excel(file_path)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Could not read file into DataFrame: {e}")

def _setup_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # Basic arguments
    # parser.add_argument(
    #     "--data", 
    #     type=convert_to_df,  # Passes the string path through your function
    #     required=True, 
    #     help="Path to the CSV file"
    # )

    parser.add_argument(
        "--data", 
        type=str,  # Passes the string path through your function
        required=True, 
        help="Path to the CSV file"
    )

    parser.add_argument(
        "--name_of_label_column", 
        type=str,  # Passes the string path through your function
        required=True, 
        help="Name of the Label Column"
    )

    parser.add_argument(
        "--name_of_message_column", 
        type=str,  # Passes the string path through your function
        required=True, 
        help="Name of the Message Column"
    )

    # parser.add_argument(
    #     "--num_workers",
    #     type=int,
    #     default=0,
    #     help="Number of workers for the data module"
    #     + " Default is 0.",
    # )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for data module"
        + " Default is 64.",
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=2,
        help="Max epoch for trainer"
        + " Default is 2.",
    )

    parser.add_argument(
        "--accelerator", 
        type=str,  # Passes the string path through your function
        default="auto",
        help="Accelerator for the trainer"
    )

    parser.add_argument(
        "--devices", 
        type=str,  # Passes the string path through your function
        default="auto",
        help="Devices for the trainer"
    )

    parser.add_argument(
        "--mlflow_tracking_uri", 
        type=str,  # Passes the string path through your function
        default="",
        help="Tracking uri to be used by mlflow"
    )

    parser.add_argument(
        "--distributed_processing",
        action="store_true",
        default=False,
        help=(
            "Will run distributed processing if used."
            "Note that it looks like there's an issue with distributed processing if mlflow is used with an sqlite db so training may crash as a result."
            "Use this only if mlflow is running a proper database."
        ),
    )


    # to call
    # df = args.data  # This is now a fully loaded pandas DataFrame object!

    return parser

@rank_zero_only
def log_artifacts_manual(mlflow_client, run_id, folder_path, trainer):
    data = trainer.datamodule
    vocab_pt_file_path = folder_path + "vocab.pt"
    sms_spam_checkpoint_file_path = folder_path + "sms_spam.ckpt"

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

                    # vocab_pt_file_path = folder_path + "vocab.pt"
                    # sms_spam_checkpoint_file_path = folder_path + "sms_spam.ckpt"

                    # torch.save(data.vocab, vocab_pt_file_path)
                    # trainer.save_checkpoint(sms_spam_checkpoint_file_path)

                    # print(data.vocab)

                    print("calling on save artifacts")

                    # mlflow_client.log_artifacts(
                    #     run_id=run_id, 
                    #     local_dir=folder_path, 
                    #     artifact_path="evaluation"
                    # )
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

def main():
    print("call main")
    multiprocessing.set_start_method("spawn", force=True)

    parser = _setup_parser()

    args = parser.parse_args()

    df = pd.DataFrame()

    mlflow_logger = None

    print(early_stop_acc)

    # callbacks = [early_stop_acc]

    callbacks = [early_stop_loss]

    if args.mlflow_tracking_uri:
        if (len(args.mlflow_tracking_uri)):
            mlflow.pytorch.autolog()
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)

            mlflow_logger = MLFlowLogger(
                experiment_name="spam_training",
                tracking_uri=args.mlflow_tracking_uri,  # Point to your local or remote server
                log_model='all'
            )
            callbacks.append(LogArtifactsCallback())

    print(callbacks)
    if args.data:
        df = convert_to_df(args.data)
        label_column = ''
        message_column = ''

        if args.name_of_label_column:
            label_column = args.name_of_label_column

        if args.name_of_message_column:
            message_column = args.name_of_message_column

        if (len(label_column) > 0 and len(message_column) > 0):

            df = df[[label_column, message_column]]

            df = df.rename(columns={label_column: "label", message_column: "message"})

    if (not df.empty):

        # print(df)

        # num_workers = args.num_workers

        num_workers = 0

        batch_size = args.batch_size
        max_epochs = args.max_epochs
        accelerator = args.accelerator
        devices = args.devices

        data = SMSDataModule(
            dataframe=df,
            batch_size=batch_size,
            num_workers=num_workers
        )


        data.setup()

        print("<------------------------data load complete-------------------------------->")

        model = SpamClassifier(
            vocab_size=len(data.vocab)
        )

        print("<------------------------model load complete-------------------------------->")

        gpus = int(torch.cuda.is_available())

        if (args.distributed_processing):

            print("calling distributed processing")
            if(gpus > 0 and accelerator == "auto"):
                accelerator = "gpu"

            trainer = Trainer(
                max_epochs=max_epochs,
                accelerator=accelerator,
                devices=devices, 
                logger=mlflow_logger,
                callbacks=callbacks,
                strategy="ddp",           # Uses Distributed Data Parallel
                num_nodes=1,              # Set >1 for multi-machine setups
                # use_distributed_sampler=False
            )
        else:
            print("calling single processing")
            trainer = Trainer(
                max_epochs=max_epochs,
                accelerator=accelerator,
                devices=devices, 
                logger=mlflow_logger,
                callbacks=callbacks
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

        if (len(callbacks) <= 0):
            torch.save(data.vocab, "vocab.pt")
            trainer.save_checkpoint("sms_spam.ckpt")





if __name__ == "__main__":
    main()
