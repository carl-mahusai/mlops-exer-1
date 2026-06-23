
# import argparse
# import pandas as pd
# import optuna
# import mlflow

# from training.final_training import train_model
# from training.hyperparameter_search import objective
# import training.metadata.tuning as tuning_metadata

# def convert_to_df(file_path):
#     try:
#         return pd.read_csv(file_path, encoding_errors='ignore') # or pd.read_excel(file_path)
#     except Exception as e:
#         raise argparse.ArgumentTypeError(f"Could not read file into DataFrame: {e}")

# def _setup_parser():
#     parser = argparse.ArgumentParser(add_help=False)

#     # Basic arguments

#     parser.add_argument(
#         "--data", 
#         type=str,  # Passes the string path through your function
#         required=True, 
#         help="Path to the CSV file"
#     )

#     parser.add_argument(
#         "--name_of_label_column", 
#         type=str,  # Passes the string path through your function
#         required=True, 
#         help="Name of the Label Column"
#     )

#     parser.add_argument(
#         "--name_of_message_column", 
#         type=str,  # Passes the string path through your function
#         required=True, 
#         help="Name of the Message Column"
#     )

#     # parser.add_argument(
#     #     "--num_workers",
#     #     type=int,
#     #     default=0,
#     #     help="Number of workers for the data module"
#     #     + " Default is 0.",
#     # )

#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=16,
#         help="Batch size for data module"
#         + " Default is 16.",
#     )

#     parser.add_argument(
#         "--max_length",
#         type=int,
#         default=50,
#         help="Max token length used in the vocab"
#         + " Default is 50.",
#     )

#     parser.add_argument(
#         "--max_vocab_size",
#         type=int,
#         default=5000,
#         help="Max size of the vocab"
#         + " Default is 5000.",
#     )

#     parser.add_argument(
#         "--embedding_dim",
#         type=int,
#         default=64,
#         help="Size of embedding dimension"
#         + " Default is 64.",
#     )

#     parser.add_argument(
#         "--hidden_dim",
#         type=int,
#         default=64,
#         help="Size of hidden dimension"
#         + " Default is 64.",
#     )

#     parser.add_argument(
#         "--lr",
#         type=float,
#         default=1e-3,
#         help="learning rate"
#         + " Default is 1e-3.",
#     )

#     parser.add_argument(
#         "--max_epochs",
#         type=int,
#         default=2,
#         help="Max epoch for trainer"
#         + " Default is 2.",
#     )

#     parser.add_argument(
#         "--accelerator", 
#         type=str,  # Passes the string path through your function
#         default="auto",
#         help="Accelerator for the trainer"
#     )

#     parser.add_argument(
#         "--num_nodes",
#         type=int,
#         default=1,
#         help="number of nodes to train on"
#         + " Default is 1.",
#     )

#     parser.add_argument(
#         "--devices", 
#         type=str,  # Passes the string path through your function
#         default="1",
#         help="Devices to be used per training node"
#     )

#     # parser.add_argument(
#     #     "--strategy", 
#     #     type=str,  # Passes the string path through your function
#     #     default="auto",
#     #     help="Tracking uri to be used by mlflow"
#     # )

#     parser.add_argument(
#         "--mlflow_tracking_uri", 
#         type=str,  # Passes the string path through your function
#         # default="",
#         help="Tracking uri to be used by mlflow"
#     )

#     parser.add_argument(
#         "--distributed_processing",
#         action="store_true",
#         default=False,
#         help=(
#             "Will run distributed processing if used."
#             "Note that it looks like there's an issue with distributed processing if mlflow is used with an sqlite db so training may crash as a result."
#             "Use this only if mlflow is running a proper database."
#         ),
#     )

#     parser.add_argument(
#         "--optimize",
#         default=False,
#         action="store_true"
#     )

#     parser.add_argument(
#         "--optimize_and_train",
#         default=False,
#         action="store_true"
#     )


#     parser.add_argument(
#         "--n_trials",
#         type=int,
#         choices=range(5, 11), 
#         default=5
#     )

#     return parser



# def main():
#     parser = _setup_parser()

#     args = parser.parse_args()

#     df = pd.DataFrame()

#     if args.data:
#         df = convert_to_df(args.data)
#         label_column = ''
#         message_column = ''

#         if args.name_of_label_column:
#             label_column = args.name_of_label_column

#         if args.name_of_message_column:
#             message_column = args.name_of_message_column

#         if (len(label_column) > 0 and len(message_column) > 0):

#             df = df[[label_column, message_column]]

#             df = df.rename(columns={label_column: "label", message_column: "message"})

#     best = tuning_metadata.BASE_PARAMETERS

#     tuned = False

#     mlflow.pytorch.autolog()
#     mlflow.set_tracking_uri(args.mlflow_tracking_uri)
#     mlflow.system_metrics.disable_system_metrics_logging()

#     if (args.optimize_and_train or (args.optimize)):
#         study = optuna.create_study(
#             direction="minimize",
#             # storage=args.mlflow_tracking_uri,
#         )
#         study.optimize(
#             lambda trial: objective(trial, df, args),
#             n_trials=args.n_trials,
#             n_jobs=1
#         )

#         best = study.best_params

#         print(best)
#         tuned = True

#     if (args.optimize_and_train or (not args.optimize)):
#         train_model(
#             args=args,
#             df=df,
#             best=best,
#             tuned=tuned
#         )

# if __name__ == "__main__":
#     main()

# training/run_training.py

import argparse

from core.train import run_training


def parse_args():

    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--data",
    #     required=True
    # )

    parser.add_argument(
        "--data", 
        type=str,  # Passes the string path through your function
        required=True, 
        help="Path to the CSV file"
    )

    # parser.add_argument(
    #     "--name_of_label_column",
    #     required=True
    # )

    parser.add_argument(
        "--name_of_label_column", 
        type=str,  # Passes the string path through your function
        required=True, 
        help="Name of the Label Column"
    )

    # parser.add_argument(
    #     "--name_of_message_column",
    #     required=True
    # )

    parser.add_argument(
        "--name_of_message_column", 
        type=str,  # Passes the string path through your function
        required=True, 
        help="Name of the Message Column"
    )

    # parser.add_argument(
    #     "--mlflow_tracking_uri",
    #     required=False,
    #     default=""
    # )

    parser.add_argument(
        "--mlflow_tracking_uri", 
        type=str,  # Passes the string path through your function
        # default="",
        help="Tracking uri to be used by mlflow"
    )

    parser.add_argument(
        "--max_epoch",
        type=int,
        default=2,
        help="Max epoch for trainer"
        + " Default is 2.",
    )

    # parser.add_argument(
    #     "--accelerator",
    #     default="cpu"
    # )

    parser.add_argument(
        "--accelerator", 
        type=str,  # Passes the string path through your function
        default="auto",
        help="Accelerator for the trainer"
    )

    # parser.add_argument(
    #     "--devices",
    #     type=int,
    #     default=1
    # )

    parser.add_argument(
        "--devices", 
        type=str,  # Passes the string path through your function
        default="1",
        help="Devices to be used per training node"
    )

    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=8
    # )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for data module"
        + " Default is 16.",
    )

    # parser.add_argument(
    #     "--embedding_dim",
    #     type=int,
    #     default=128
    # )

    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of embedding dimension"
        + " Default is 64.",
    )

    # parser.add_argument(
    #     "--hidden_dim",
    #     type=int,
    #     default=64
    # )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Size of hidden dimension"
        + " Default is 64.",
    )

    # parser.add_argument(
    #     "--lr",
    #     type=float,
    #     default=1e-3
    # )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate"
        + " Default is 1e-3.",
    )

    # parser.add_argument(
    #     "--max_vocab_size",
    #     type=int,
    #     default=5000
    # )

    parser.add_argument(
        "--max_vocab_size",
        type=int,
        default=5000,
        help="Max size of the vocab"
        + " Default is 5000.",
    )

    # parser.add_argument(
    #     "--max_length",
    #     type=int,
    #     default=64
    # )

    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Max token length used in the vocab"
        + " Default is 50.",
    )

    # parser.add_argument(
    #     "--optimize_and_train",
    #     action="store_true"
    # )

    parser.add_argument(
        "--optimize_and_train",
        default=False,
        action="store_true"
    )


    # parser.add_argument(
    #     "--distributed_processing",
    #     action="store_true"
    # )

    # parser.add_argument(
    #     "--num_nodes",
    #     type=int,
    #     default=1
    # )

    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes to train on"
        + " Default is 1.",
    )

    # parser.add_argument(
    #     "--n_trials",
    #     type=int,
    #     default=20
    # )

    parser.add_argument(
        "--n_trials",
        type=int,
        choices=range(5, 11), 
        default=5
    )

    return parser.parse_args()


def main():

    args = parse_args()

    run_training(args)


if __name__ == "__main__":
    main()