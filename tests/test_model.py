import sys

sys.path.append(".")  # add root of project to path

import torch
import onnxruntime as ort

from model.simple_mlp import SimpleMLP
from configs.args import ModelArgs

model = SimpleMLP(ModelArgs())

OUT_SHAPE = (1, 10)


def test_forward_pass():
    x = model.dummy_input
    y = model(x)
    assert y.shape == OUT_SHAPE


def test_save_load_model(tmp_path):
    model.save_model(tmp_path / "test")
    model.from_pretrained(tmp_path / "test")
    x = model.dummy_input
    y = model(x)
    assert y.shape == OUT_SHAPE


def test_onnx(tmp_path):
    model.export_onnx(tmp_path / "test" / "model.onnx")
    ort_session = ort.InferenceSession(tmp_path / "test" / "model.onnx")
    dummy_input = model.dummy_input
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    assert ort_outs[0].shape == OUT_SHAPE
    regular_outs = model(dummy_input)
    mean_abs_error = torch.abs(regular_outs - torch.tensor(ort_outs[0])).mean()
    assert mean_abs_error < 0.1
