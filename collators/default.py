from pathlib import Path
import warnings

import torch
import torch.nn.functional as F
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
        self.mask_proportion_tolerance = args.mask_proportion_tolerance

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
            if (
                all(
                    [
                        audio_path.with_suffix(".pitch.pt").exists(),
                        audio_path.with_suffix(".energy.pt").exists(),
                        audio_path.with_suffix(".vad.pt").exists(),
                        audio_path.with_suffix(".pad_mask.pt").exists(),
                    ]
                )
                and not self.overwrite
            ):
                result["pitch"].append(torch.load(audio_path.with_suffix(".pitch.pt")))
                result["energy"].append(
                    torch.load(audio_path.with_suffix(".energy.pt"))
                )
                result["vad"].append(torch.load(audio_path.with_suffix(".vad.pt")))
                result["mask_pad"].append(
                    torch.load(audio_path.with_suffix(".pad_mask.pt"))
                )
            else:
                audio, sr = librosa.load(item["audio"], sr=22050)
                # if shorter than max_length * 256, pad
                if len(audio) < self.max_length * 256:
                    # pad mask
                    pad_mask = torch.zeros(self.max_length)
                    pad_mask[: len(audio) // 256 + 1] = 1
                    result["mask_pad"].append(pad_mask)
                    audio = np.pad(audio, (0, self.max_length * 256 - len(audio)))
                # if longer than max_length * 256, get random window
                elif len(audio) > self.max_length * 256:
                    start = np.random.randint(0, len(audio) - self.max_length * 256)
                    audio = audio[start : start + self.max_length * 256]
                    # pad mask
                    pad_mask = torch.zeros(self.max_length)
                    pad_mask[:] = 1
                    result["mask_pad"].append(pad_mask)
                result["audio"].append(audio)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vocex_result = self.vocex(audio, sr)
                pitch = vocex_result["measures"]["pitch"][0, :-1]
                energy = vocex_result["measures"]["energy"][0, :-1]
                vad = vocex_result["measures"]["voice_activity_binary"][0, :-1]
                # min-max normalize
                pitch = (pitch - pitch.min()) / (pitch.max() - pitch.min())
                energy = (energy - energy.min()) / (energy.max() - energy.min())
                vad = (vad - vad.min()) / (vad.max() - vad.min())
                pitch[torch.isnan(pitch)] = 0
                energy[torch.isnan(energy)] = 0
                vad[torch.isnan(vad)] = 0
                result["pitch"].append(pitch)
                result["energy"].append(energy)
                result["vad"].append(vad)
                # save
                torch.save(pitch, audio_path.with_suffix(".pitch.pt"))
                torch.save(energy, audio_path.with_suffix(".energy.pt"))
                torch.save(vad, audio_path.with_suffix(".vad.pt"))
                torch.save(pad_mask, audio_path.with_suffix(".pad_mask.pt"))
            # bucketize
            pitch = result["pitch"][-1]
            energy = result["energy"][-1]
            vad = result["vad"][-1]
            # 1 is reserved for masking, 0 is reserved for padding
            pitch = torch.bucketize(pitch, self.bins) + 2
            energy = torch.bucketize(energy, self.bins) + 2
            vad = torch.bucketize(vad, self.bins) + 2
            result["pitch"][-1] = pitch
            result["energy"][-1] = energy
            result["vad"][-1] = vad
            # mask
            # We always mask a fixed proportion of the frames, but vary the length of the masked spans.
            # We mask self.mask_p of the frames, but vary the length of the masked spans (using self.mask_l)
            mask = torch.ones_like(result["mask_pad"][-1]).bool()
            pad_mask = result["mask_pad"][-1].bool()
            while (mask.bool() & pad_mask).sum() / pad_mask.sum() > self.mask_p:
                start = np.random.randint(0, self.max_length)
                new_mask = mask.clone()
                new_mask[start : start + self.mask_l] = 0
                if (
                    new_mask.bool() & pad_mask
                ).sum() / pad_mask.sum() >= self.mask_p - self.mask_proportion_tolerance:
                    mask = new_mask
            result["pitch_masked"].append(result["pitch"][-1])
            result["pitch_masked"][-1] *= ~mask
            result["pitch_masked"][-1] += ~mask
            result["energy_masked"].append(result["energy"][-1])
            result["energy_masked"][-1] *= ~mask
            result["energy_masked"][-1] += ~mask
            result["vad_masked"].append(result["vad"][-1])
            result["vad_masked"][-1] *= ~mask
            result["vad_masked"][-1] += ~mask
            result["mask_pred"].append(mask)
        # stack
        result["pitch"] = torch.stack(result["pitch"]).long()
        result["energy"] = torch.stack(result["energy"]).long()
        result["vad"] = torch.stack(result["vad"]).long()
        result["pitch_masked"] = torch.stack(result["pitch_masked"]).long()
        result["energy_masked"] = torch.stack(result["energy_masked"]).long()
        result["vad_masked"] = torch.stack(result["vad_masked"]).long()
        result["mask_pad"] = torch.stack(result["mask_pad"]).bool()
        result["mask_pred"] = torch.stack(result["mask_pred"]).bool()
        return result
