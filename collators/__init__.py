import yaml

from .default import MNISTCollator
from configs.args import CollatorArgs
import torch


def get_collator(args: CollatorArgs):
    return {
        "default": MNISTCollator,
    }[
        args.name
    ](args)
