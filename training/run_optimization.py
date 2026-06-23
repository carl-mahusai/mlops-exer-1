import pandas as pd

from core.tune import run_hyperparameter_search

from .parser_util import _setup_parser

def main():

    parser = _setup_parser()

    args = parser.parse_args()

    run_hyperparameter_search(args, pd.DataFrame())


if __name__ == "__main__":
    main()