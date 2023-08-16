from pathlib import Path
from collections import OrderedDict
import os

import yaml
import torch
from torch import nn
from transformers.utils.hub import cached_file
from rich.console import Console

console = Console()

from configs.args import ModelArgs


class MLPLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: nn.Module = nn.GELU(),
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x):
        res_x = x
        x = self.linear(x)
        if self.residual:
            x = x + res_x
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SimpleMLP(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
    ):
        super().__init__()

        self.in_layer = nn.Sequential(
            OrderedDict(
                [
                    ("layer_in_linear", nn.Linear(28 * 28, args.hidden_dim)),
                    ("layer_in_gelu", nn.GELU()),
                ]
            )
        )

        self.hidden_layers = nn.ModuleList(
            [
                MLPLayer(
                    args.hidden_dim,
                    args.hidden_dim,
                    activation=nn.GELU(),
                    dropout=args.dropout,
                    residual=args.residual,
                )
                for _ in range(args.n_layers)
            ]
        )

        self.out_layer = nn.Linear(args.hidden_dim, 10)

        self.args = args

    def forward(self, x):
        x = self.in_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.out_layer(x)
        return x

    def save_model(self, path, accelerator=None, onnx=False):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if accelerator is not None:
            accelerator.save_model(self, path)
        else:
            torch.save(self.state_dict(), path / "pytorch_model.bin")
        if onnx:
            try:
                self.export_onnx(path / "model.onnx")
            except Exception as e:
                console.print(f"[red]Skipping ONNX export[/red]: {e}")
        with open(path / "model_config.yml", "w") as f:
            f.write(yaml.dump(self.args.__dict__, Dumper=yaml.Dumper))

    @staticmethod
    def from_pretrained(path_or_hubid):
        path = Path(path_or_hubid)
        if path.exists():
            config_file = path / "model_config.yml"
            model_file = path / "pytorch_model.bin"
        else:
            config_file = cached_file(path_or_hubid, "model_config.yml")
            model_file = cached_file(path_or_hubid, "pytorch_model.bin")
        args = yaml.load(open(config_file, "r"), Loader=yaml.Loader)
        args = ModelArgs(**args)
        model = SimpleMLP(args)
        model.load_state_dict(torch.load(model_file))
        return model

    @property
    def dummy_input(self):
        torch.manual_seed(0)
        return torch.randn(1, 28 * 28)

    def export_onnx(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            self,
            self.dummy_input,
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=11,
        )
