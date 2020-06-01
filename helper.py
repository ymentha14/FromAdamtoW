import argparse


def parse_arguments():
    """Parses the global parameters from the command line arguments."""

    parser = argparse.ArgumentParser(description="From Adam to AdamW")

    parser.add_argument(
        "--task_name",
        default="all",
        help="By default execute all tasks. When specified, execute a specific task. Valid values: text_cls, speech_cls, imgages_cls.",
    )

    parser.add_argument(
        "--optimizer",
        default="all",
        help="By default experiment with all optimizers. When specified, execute a specific optimizer. Valid values: Adam, AdamW, SGD or 'all'.",
    )

    parser.add_argument(
        "--param_file",
        default="param.json",
        help="Parameter files containing all different settings.",
    )

    parser.add_argument(
        "--learning_rate",
        help="Learning rate. If not specified, the model will use the best learning rate found during cross-validation.",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--max_cross_validation_epochs",
        help=f"Maximal number of epochs to use in cross-validation training. Default 100.",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--num_epochs",
        help="Number of epochs to train. If not specified, the model will use the best number found during cross-validation.",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--num_runs",
        help=f"Number of independent execution runs. Default 10.",
        default=10,
    )

    parser.add_argument(
        "--cross_validation",
        help="For each model, find and display the best learning rate and optimal epoch using cross-validation.",
        action="store_true",
    )

    parser.add_argument(
        "--verbose",
        help="Print the loss during training. Should not be activated when testing for execution time.",
        action="store_true",
    )

    return parser.parse_args()
