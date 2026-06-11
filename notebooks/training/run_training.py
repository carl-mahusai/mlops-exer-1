import argparse
import torch
import pandas as pd

from spam_checker.data.spam_lit_datamodule import SMSDataModule
from spam_checker.models.spam_classifier import SpamClassifier
from lightning import Trainer

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

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the data module"
        + " Default is 0.",
    )

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
        default=10,
        help="Max epoch for trainer"
        + " Default is 10.",
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


    # to call
    # df = args.data  # This is now a fully loaded pandas DataFrame object!

    return parser

def main():
    print("call main")
    parser = _setup_parser()

    args = parser.parse_args()

    df = pd.DataFrame()

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

        print(df)

        num_workers = args.num_workers

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

        print(len(data.vocab))

        model = SpamClassifier(
            vocab_size=len(data.vocab)
        )

        trainer = Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=devices)

        trainer.fit(
            model,
            datamodule=data,
        )

        trainer.test(
            model,
            datamodule=data,
        )

        torch.save(data.vocab, "vocab.pt")
        trainer.save_checkpoint("sms_spam.ckpt")



if __name__ == "__main__":
    main()
