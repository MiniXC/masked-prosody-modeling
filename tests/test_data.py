import sys

sys.path.append(".")  # add root of project to path

from datasets import load_dataset
from torch.utils.data import DataLoader

from configs.args import ModelArgs, TrainingArgs, CollatorArgs
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

IN_SHAPE = (1, 28 * 28)
OUT_SHAPE = (1, 10)


def test_dataloader():
    for batch in dataloader:
        assert batch["image"].shape == (default_args.batch_size, 28 * 28)
        assert batch["target"].shape == (default_args.batch_size,)
        assert batch["target_onehot"].shape == (default_args.batch_size, 10)
        assert batch["image"].max() <= 1
        assert batch["image"].min() >= 0
        assert batch["target"].max() <= 9
        assert batch["target"].min() >= 0
        break
