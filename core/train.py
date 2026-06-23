import pandas as pd

from training.hyperparameter_search import hyperparameter_search
from training.final_training import train_model
from core.tune import run_hyperparameter_search
from core.data import read_df


def run_training(args):

    # dataframe = pd.read_csv(args.data)
    dataframe = read_df(args)

    if dataframe.empty:
        raise ValueError("Dataframe is empty")

    if args.optimize_and_train:

        # best_params = hyperparameter_search(
        #     dataframe=dataframe,
        #     args=args
        # )

        best_params = run_hyperparameter_search(args, dataframe)

        args.batch_size = best_params["batch_size"]
        args.embedding_dim = best_params["embedding_dim"]
        args.hidden_dim = best_params["hidden_dim"]
        args.lr = best_params["lr"]
        args.max_vocab_size = best_params["max_vocab_size"]
        args.max_length = best_params["max_length"]

    train_model(
        args=args,
        dataframe=dataframe
    )