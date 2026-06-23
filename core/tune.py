# core/tune.py

import pandas as pd

from training.hyperparameter_search import (
    hyperparameter_search
)


def run_hyperparameter_search(args, dataframe):

    # dataframe = pd.read_csv(args.data)

    if dataframe.empty:
        raise ValueError("Dataframe is empty")

    best_params = hyperparameter_search(
        dataframe=dataframe,
        args=args
    )

    return best_params