from pathlib import Path
import warnings

import torch
from vocex import Vocex
import librosa
import numpy as np

from configs.args import CollatorArgs


class LibriTTSCollator:
    def __init__(self, args: CollatorArgs):
        self.overwrite = args.overwrite
        self.bins = torch.linspace(0, 1, args.bin_size)
        self.vocex = Vocex.from_pretrained(args.vocex_model)
        self.mask_p = args.mask_proportion
        self.mask_l = args.mask_length
        self.max_length = args.max_length

    def __call__(self, batch):
        result = {
            "audio": [],
            "pitch": [],
            "energy": [],
            "vad": [],
            "pitch_masked": [],
            "energy_masked": [],
            "vad_masked": [],
            "mask_pad": [],
            "mask_pred": [],
        }
        for i, item in enumerate(batch):
            audio_path = Path(item["audio"])
            if audio_path.with_suffix(".pitch.pt").exists() and not self.override:
                result["pitch"].append(torch.load(audio_path.with_suffix(".pitch.pt")))
                result["energy"].append(
                    torch.load(audio_path.with_suffix(".energy.pt"))
                )
                result["vad"].append(torch.load(audio_path.with_suffix(".vad.pt")))
                continue
            audio, sr = librosa.load(item["audio"], sr=22050)
            # if shorter than max_length * 256, pad
            if len(audio) < self.max_length * 256:
                # pad mask
                pad_mask = torch.zeros(self.max_length + 1)
                pad_mask[: len(audio) // 256 + 1] = 1
                result["mask_pad"].append(pad_mask)
                audio = np.pad(audio, (0, self.max_length * 256 - len(audio)))
            # if longer than max_length * 256, get random window
            elif len(audio) > self.max_length * 256:
                start = np.random.randint(0, len(audio) - self.max_length * 256)
                audio = audio[start : start + self.max_length * 256]
            result["audio"].append(audio)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vocex_result = self.vocex(audio, sr)
            # min-max normalize
            pitch = (pitch - pitch.min()) / (pitch.max() - pitch.min())
            energy = (energy - energy.min()) / (energy.max() - energy.min())
            vad = (vad - vad.min()) / (vad.max() - vad.min())
            result["pitch"].append(pitch)
            result["energy"].append(energy)
            result["vad"].append(vad)
            # bucketize
            pitch = (
                torch.bucketize(pitch, self.bins) + 2
            )  # 1 is reserved for masking, 0 is reserved for padding
            energy = torch.bucketize(energy, self.bins) + 2
            vad = torch.bucketize(vad, self.bins) + 2
            # save
            torch.save(pitch, audio_path.with_suffix(".pitch.pt"))
            torch.save(energy, audio_path.with_suffix(".energy.pt"))
            torch.save(vad, audio_path.with_suffix(".vad.pt"))
        # stack
        result["pitch"] = torch.stack(result["pitch"])
        result["energy"] = torch.stack(result["energy"])
        result["vad"] = torch.stack(result["vad"])
        # mask
        # We always mask a fixed proportion of the frames, but vary the length of the masked spans.
        # We mask self.mask_p of the frames, but vary the length of the masked spans (using self.mask_l)
        mask = torch.ones_like(result["pitch"])
        while (mask & pad_mask).sum() / pad_mask.sum() < self.mask_p:
            start = np.random.randint(0, pad_mask.sum() - self.mask_l)
            mask[start : start + self.mask_l] = 1
        result["pitch_masked"] = result["pitch"].clone()
        result["pitch_masked"][mask == 1] = 1
        result["energy_masked"] = result["energy"].clone()
        result["energy_masked"][mask == 1] = 1
        result["vad_masked"] = result["vad"].clone()
        result["vad_masked"][mask == 1] = 1
        result["mask_pred"] = mask
        return result
