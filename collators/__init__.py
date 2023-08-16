import yaml

from .default import LibriTTSCollator
from configs.args import CollatorArgs


def get_collator(args: CollatorArgs):
    return {
        "default": LibriTTSCollator,
    }[
        args.name
    ](args)
