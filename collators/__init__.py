import yaml

from .default import LibriTTSCollator, LibriTTSAlgoCollator, FischerCollator
from configs.args import CollatorArgs


def get_collator(args: CollatorArgs):
    return {
        "default": LibriTTSCollator,
        "algo": LibriTTSAlgoCollator,
        "fischer": FischerCollator,
    }[
        args.name
    ](args)
