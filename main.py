import __main__
import os
import sys
import logging
import argparse
import subprocess
from multiprocessing import get_context
from encoder.trainer.train import run
from encoder.utils.config import *

logging.root.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    p_train = subparsers.add_parser("train", help="Start training.")

    p_validate = subparsers.add_parser("validate", help="Start validating.")

    p_test = subparsers.add_parser("test", help="Start testing.")

    p_evaluate_model = subparsers.add_parser(
        "evaluate_model", help="Start evaluating model."
    )

    p_train.add_argument(
        "--config", type=str, required=True, help="Path of the config file.",
    )

    p_train.add_argument(
        "--stage", type=int, default=None, help="Stage number to run.",
    )

    p_validate.add_argument(
        "--config", type=str, required=True, help="Path of the config file.",
    )

    p_validate.add_argument(
        "--stage", type=int, default=None, help="Stage number to run.",
    )

    p_test.add_argument(
        "--config", type=str, required=True, help="Path of the config file.",
    )

    p_test.add_argument(
        "--stage", type=int, default=None, help="Stage number to run.",
    )

    p_evaluate_model.add_argument(
        "--config", type=str, required=True, help="Path of the config file.",
    )

    p_evaluate_model.add_argument(
        "--stage", type=int, default=None, help="Stage number to run.",
    )

    p_generate = subparsers.add_parser(
        "generate", help="Generate an example configuration."
    )

    p_generate.add_argument(
        "--stages",
        type=str,
        required=True,
        help="Stages to execute. Example: qa,qa,kb_encode",
    )
    p_generate.add_argument(
        "--print", action="store_true", help="Direct config output to screen."
    )
    p_generate.add_argument(
        "--output",
        type=str,
        default="config.json",
        help="JSON config file output path.",
    )

    args = parser.parse_args()
    if args.command in ("train", "validate", "test", "evaluate_model"):
        config = load_config(args.config)
        assert len(config.stages) == len(
            config.configs
        ), "Pipeline stage number must be equal to the number of stage configs."

        # Copied from pytorch lightning ddp plugin
        if args.stage is None:
            # Check if the current calling command looked like
            # `python a/b/c.py` or `python -m a.b.c`
            # See https://docs.python.org/3/reference/import.html#main-spec
            if __main__.__spec__ is None:  # pragma: no-cover
                # pull out the commands used to run the script and
                # resolve the abs file path
                command = sys.argv
                full_path = os.path.abspath(command[0])

                command[0] = full_path
                # use the same python interpreter and actually running
                command = [sys.executable] + command
            else:  # Script called as `python -m a.b.c`
                command = [sys.executable, "-m", __main__.__spec__.name] + sys.argv[1:]

            for i in range(len(config.stages)):
                logging.info(f"Running stage {i} type: {config.stages[i]}")
                logging.info("=" * 100)
                process = subprocess.Popen(command + ["--stage", str(i)])
                process.wait()
        else:
            assert (
                0 <= args.stage < len(config.stages)
            ), f"Stage number {args.stage} out of range."
            run(config, args.stage, mode=args.command)

    elif args.command == "generate":
        generate_config(args.stages.split(","), args.output, args.print)
