from prefect import task

from core.train import run_training


@task
def run_training_task(args):

    run_training(args)