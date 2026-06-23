from prefect import task
import pandas as pd

from core.tune import run_hyperparameter_search


@task
def run_tuning_task(args):

    run_hyperparameter_search(args, pd.DataFrame())