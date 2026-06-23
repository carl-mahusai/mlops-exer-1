import argparse

from core.train import run_training


def _setup_parser():

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
        choices=range(1, 11), 
        default=1
    )

    return parser


def main():

    parser = _setup_parser()

    args = parser.parse_args()

    print(args)

    # run_training(args)

    print(args.data)

    run_training(args)


if __name__ == "__main__":
    main()