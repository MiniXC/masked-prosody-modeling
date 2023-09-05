import yaml

from .default import LibriTTSCollator, LibriTTSAlgoCollator
from configs.args import CollatorArgs


def get_collator(args: CollatorArgs):
    return {
        "default": LibriTTSCollator,
        "algo": LibriTTSAlgoCollator,
    }[
        args.name
    ](args)
