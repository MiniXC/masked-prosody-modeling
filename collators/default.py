import torch
from torch import nn
import numpy as np

from configs.args import CollatorArgs


class MNISTCollator(nn.Module):
    def __init__(
        self,
        args: CollatorArgs,
    ):
        super().__init__()

        self.args = args

    def __call__(self, batch):
        if self.args.normalize:
            x = torch.tensor(
                np.array([np.asarray(b["image"]).flatten() / 255 for b in batch])
            ).to(torch.float32)
        else:
            x = torch.tensor(
                np.array([np.asarray(b["image"]).flatten() for b in batch])
            ).to(torch.float32)
        y = torch.tensor([b["label"] for b in batch]).long()
        if self.args.onehot:
            y_onehot = torch.nn.functional.one_hot(y, num_classes=10).to(torch.float32)
            return {
                "image": x,
                "target": y,
                "target_onehot": y_onehot,
            }
        else:
            return {
                "image": x,
                "target": y,
            }
