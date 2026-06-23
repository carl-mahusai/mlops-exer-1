from core.train import run_training

from .parser_util import _setup_parser

def main():

    parser = _setup_parser()

    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()