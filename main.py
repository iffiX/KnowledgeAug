import __main__
import os
import sys
import logging
import argparse
import subprocess
from encoder.trainer.train import run, export_model, export_bert_input, export_t5_input
from encoder.utils.config import *

logging.root.setLevel(logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", help="Attach to a remote PyCharm debug server", action="store_true"
    )
    parser.add_argument(
        "--debug_ip", help="PyCharm debug server IP", type=str, default="localhost"
    )
    parser.add_argument(
        "--debug_port", help="PyCharm debug server port", type=int, default=80
    )

    subparsers = parser.add_subparsers(dest="command")

    p_train = subparsers.add_parser("train", help="Start training.")

    p_validate = subparsers.add_parser("validate", help="Start validating.")

    p_test = subparsers.add_parser("test", help="Start testing.")

    p_evaluate_model = subparsers.add_parser(
        "evaluate_model", help="Start evaluating model."
    )

    p_export_model = subparsers.add_parser(
        "export_model", help="Start exporting model."
    )

    p_export_t5_input = subparsers.add_parser(
        "export_t5_input", help="Start exporting inputs for T5 model."
    )

    p_export_bert_input = subparsers.add_parser(
        "export_bert_input", help="Start exporting inputs for BERT like model."
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

    p_export_model.add_argument(
        "--config", type=str, required=True, help="Path of the config file.",
    )

    p_export_model.add_argument(
        "--stage", type=int, required=True, help="Stage number to run.",
    )

    p_export_model.add_argument(
        "--path", type=str, required=True, help="Save destination.",
    )

    p_export_t5_input.add_argument(
        "--config", type=str, required=True, help="Path of the config file.",
    )

    p_export_t5_input.add_argument(
        "--stage", type=int, required=True, help="Stage number to run.",
    )

    p_export_t5_input.add_argument(
        "--path", type=str, required=True, help="Save destination.",
    )

    p_export_bert_input.add_argument(
        "--config", type=str, required=True, help="Path of the config file.",
    )

    p_export_bert_input.add_argument(
        "--stage", type=int, required=True, help="Stage number to run.",
    )

    p_export_bert_input.add_argument(
        "--path", type=str, required=True, help="Save destination.",
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

            logging.info(f"Total stage num: {len(config.stages)}")
            for i in range(len(config.stages)):
                if not config.configs[i].skip:
                    logging.info(f"Running stage {i} type: {config.stages[i]}")
                    logging.info("=" * 100)
                    process = subprocess.Popen(command + ["--stage", str(i)])
                    process.wait()
                else:
                    logging.info(f"Skipping stage {i} type: {config.stages[i]}")
                    logging.info("=" * 100)
        else:
            assert (
                0 <= args.stage < len(config.stages)
            ), f"Stage number {args.stage} out of range."

            if args.debug:
                logging.info("Debug server enabled")
                import pydevd_pycharm

                pydevd_pycharm.settrace(
                    args.debug_ip,
                    port=args.debug_port,
                    stdoutToServer=True,
                    stderrToServer=True,
                )
            run(config, args.stage, mode=args.command)

    elif args.command == "export_model":
        config = load_config(args.config)
        export_model(config, args.stage, args.path)

    elif args.command == "export_bert_input":
        config = load_config(args.config)
        export_bert_input(config, args.stage, args.path)

    elif args.command == "export_t5_input":
        config = load_config(args.config)
        export_t5_input(config, args.stage, args.path)

    elif args.command == "generate":
        generate_config(args.stages.split(","), args.output, args.print)
