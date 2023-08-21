import sys

sys.path.append(".")  # add root of project to path

import torch
import onnxruntime as ort

from model.masked_prosody_model import MaskedProsodyModel
from configs.args import ModelArgs, CollatorArgs

collator_args = CollatorArgs()
model_args = ModelArgs()

model_args.bins = collator_args.bin_size
model_args.max_length = collator_args.max_length

model = MaskedProsodyModel(model_args)

OUT_SHAPE = (1, 3, model_args.max_length, model_args.bins)


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
