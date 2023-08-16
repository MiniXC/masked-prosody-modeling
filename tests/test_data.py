import sys

sys.path.append(".")  # add root of project to path

from datasets import load_dataset
from torch.utils.data import DataLoader

from configs.args import TrainingArgs, CollatorArgs
from collators import get_collator

default_args = TrainingArgs()
default_collator_args = CollatorArgs()

train_dataset = load_dataset(default_args.dataset, split=default_args.train_split)
val_dataset = load_dataset(default_args.dataset, split=default_args.val_split)

collator = get_collator(default_collator_args)

dataloader = DataLoader(
    train_dataset,
    batch_size=default_args.batch_size,
    shuffle=True,
    collate_fn=collator,
)


def test_dataloader():
    for batch in dataloader:
        print(batch)
        break
