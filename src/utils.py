import argparse


def get_run_args(*args_list):
    parser = argparse.ArgumentParser()
    parsed_args = []

    for arg, default in args_list:
        parser.add_argument(f'-{arg}', type=type(default), default=default)

    args = parser.parse_args()

    for arg, _ in args_list:
        parsed_args.append(getattr(args, arg))

    return parsed_args
