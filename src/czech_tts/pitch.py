"""Stolen from
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/data_function.py."""

from typing import Optional
import torch
import librosa
import torch.nn.functional as F
import numpy as np


def estimate_pitch(
    wav: torch.Tensor,
    mel_window: int,
    mel_len: int,
    normalize_mean: Optional[torch.Tensor] = None,
    normalize_std: Optional[torch.Tensor] = None,
):
    pitch_mel, _, _ = librosa.pyin(
        wav.numpy()[0],
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        frame_length=mel_window,
    )
    assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0

    pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
    pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
    pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))

    pitch_mel = pitch_mel.float()

    if normalize_mean is not None:
        assert normalize_std is not None
        pitch_mel = normalize_pitch(pitch_mel, normalize_mean, normalize_std)

    return pitch_mel


def normalize_pitch(pitch, mean, std):
    zeros = pitch == 0.0
    pitch -= mean[:, None]
    pitch /= std[:, None]
    pitch[zeros] = 0.0
    return pitch
