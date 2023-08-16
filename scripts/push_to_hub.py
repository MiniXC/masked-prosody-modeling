from argparse import ArgumentParser
import os

from util.remote import push_to_hub


def main():
    parser = ArgumentParser(
        description="Push a checkpoint directory to the huggingface model hub."
    )
    parser.add_argument("checkpoint_path", type=str, default=None)
    parser.add_argument("repository", type=str, default=None)

    args = parser.parse_args()
    push_to_hub(args.repository, args.checkpoint_path)


if __name__ == "__main__":
    main()
