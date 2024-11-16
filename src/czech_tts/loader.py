import os
from typing import Iterable, NamedTuple, TypedDict
import pandas as pd
import numpy as np
from praatio.utilities.utils import Interval
from torchaudio import load
import torchaudio
from praatio.textgrid import openTextgrid
from tqdm.auto import tqdm
import torch

from czech_tts.pitch import estimate_pitch


class Recording(TypedDict):
    id: int
    waveform: np.ndarray
    speaker_id: str
    text: str


class CommonVoiceLoader:
    """Loads CommonVoice Dataset."""

    def __init__(self, root_path: str, split: str, sr: int = 16000) -> None:
        self.root = root_path
        self.split = split
        self._tsv = None
        self.sr = sr

    @property
    def tsv(self) -> pd.DataFrame:
        if self._tsv is None:
            self._tsv = pd.read_csv(
                os.path.join(self.root, "cs", f"{self.split}.tsv"), sep="\t"
            )

        return self._tsv

    def __len__(self) -> int:
        return self.tsv.shape[0]

    def __iter__(self) -> Iterable[Recording]:
        for idx, row in self.tsv.iterrows():
            yield {
                "id": idx,
                "waveform": self.load_waveform(row["path"]),
                "text": row["sentence"],
                "speaker_id": row["client_id"],
            }

    def load_waveform(self, clip_path: str) -> np.ndarray:
        wav, sr = load(
            os.path.join(self.root, "cs", "clips", clip_path),
            format="mp3",
            backend="ffmpeg",
        )

        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        return wav.numpy()[0]

    def dump_mfa(self, mfa_root_path: str) -> None:
        os.makedirs(mfa_root_path, exist_ok=True)
        for idx, row in tqdm(
            self.tsv.iterrows(), desc="Generating files", total=len(self)
        ):
            speaker_id = row["client_id"]
            waveform = self.load_waveform(row["path"])
            text = row["sentence"]
            name, _ = os.path.splitext(row["path"])

            speaker_root = os.path.join(mfa_root_path, speaker_id)
            os.makedirs(speaker_root, exist_ok=True)
            torchaudio.save(
                os.path.join(speaker_root, f"{name}.wav"),
                torch.from_numpy(waveform).unsqueeze(0),
                self.sr,
                channels_first=True,
                format="wav",
            )
            with open(os.path.join(speaker_root, f"{name}.txt"), mode="w") as txt_file:
                print(text, file=txt_file, end="")


class TrainRecording(TypedDict):
    id: int
    mel: torch.Tensor
    speaker_id: int
    tokens: torch.Tensor
    pitches: torch.Tensor
    durations: torch.Tensor


class RecordingAlignment(NamedTuple):
    durations: torch.Tensor
    tokens: torch.Tensor
    start: float
    end: float


class TrainLoader:
    """Loads dataset aligned with
    [MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html)"""

    def __init__(
        self,
        root_path: str,
        split: str,
        sr: int = 16000,
        mel_window: int = 1024,
        mel_channels: int = 80,
        mel_hop_length: int = 256,
    ) -> None:
        self.root = root_path
        self.split = split
        self.sr = sr
        self._tsv = None
        self._vocab_dict = {" ": 0}
        self._speakers = {}
        self._mel_kwargs = {
            "n_fft": mel_window,
            "window_length": mel_window,
            "hop_length": mel_hop_length,
            "n_mels": mel_channels,
        }
        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, **self._mel_kwargs
        )

    @property
    def tsv(self) -> pd.DataFrame:
        if self._tsv is None:
            self._tsv = pd.read_csv(
                os.path.join(self.root, "cs", f"{self.split}.tsv"), sep="\t"
            )

        return self._tsv

    def __len__(self) -> int:
        return self.tsv.shape[0]

    def __iter__(self) -> Iterable[TrainRecording]:
        for idx, row in self.tsv.iterrows():
            name, _ = os.path.splitext(row["path"])
            client_id = row["client_id"]
            assert isinstance(client_id, str)

            wav = self.load_waveform(client_id, name)
            speaker_id = self.get_speaker_id(client_id)
            alignment = self.load_alignment(client_id, name)
            wav_cliped = self.clip_waveform(wav, alignment.start, alignment.end)
            mel = self.get_mel_spectogram(wav_cliped)

            # TODO: Does mel frames equal sum of durations?
            pitches = self.get_pitches(wav_cliped, mel.shape[1])

            # TODO: What is the format of pitches?

            assert isinstance(idx, int)
            yield {
                "mel": mel,
                "id": idx,
                "tokens": alignment.tokens,
                "durations": alignment.durations,
                "speaker_id": speaker_id,
                "pitches": pitches,
            }

    def get_speaker_id(self, client_id: str) -> int:
        if client_id not in self._speakers:
            self._speakers[client_id] = len(self._speakers)

        return self._speakers[client_id]

    def load_waveform(self, client_id: str, name: str) -> torch.Tensor:
        wav, sr = load(
            os.path.join(self.root, "cs", "mfa", client_id, f"{name}.wav"),
            format="wav",
            backend="ffmpeg",
        )

        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        return wav

    def get_token(self, phoneme: str) -> int:
        if phoneme not in self._vocab_dict:
            self._vocab_dict[phoneme] = len(self._vocab_dict)

        return self._vocab_dict[phoneme]

    def secs_to_mel_frames(
        self, seconds: float, round_delta: float = 0.0
    ) -> tuple[int, float]:
        frame_secs = self._mel_kwargs["window_length"] / self.sr
        frames_float = seconds / frame_secs - round_delta
        frames_round = int(round(frames_float))
        next_round_delta = frames_round - frames_float
        return frames_round, next_round_delta

    def load_alignment(self, client_id: str, name: str) -> RecordingAlignment:
        aligned = openTextgrid(
            os.path.join(self.root, "cs", "mfa", client_id, f"{name}.TextGrid"),
            includeEmptyIntervals=True,
        )
        phonesTier = aligned.getTier("phones")

        durations = []
        tokens = []
        round_delta = 0
        for entry in phonesTier.entries:
            assert isinstance(entry, Interval)
            tokens.append(self.get_token(entry.label))
            dur_secs = entry.end - entry.start
            frames, round_delta = self.secs_to_mel_frames(dur_secs, round_delta)
            durations.append(frames)

        return RecordingAlignment(
            durations=torch.tensor(durations, dtype=torch.int32),
            tokens=torch.tensor(tokens, dtype=torch.int32),
            start=phonesTier.minTimestamp,
            end=phonesTier.maxTimestamp,
        )

    def clip_waveform(
        self, wav: torch.Tensor, start: float, end: float
    ) -> torch.Tensor:
        start_ind = int(round(start * self.sr))
        end_ind = int(round(end * self.sr))
        return wav[start_ind, end_ind]

    def get_mel_spectogram(self, wav: torch.Tensor) -> torch.Tensor:
        return self._mel_transform(wav)

    def get_pitches(self, waveform: torch.Tensor, mel_frames: int) -> torch.Tensor:
        return estimate_pitch(
            waveform,
            mel_window=self._mel_kwargs["window_length"],
            mel_len=mel_frames,
        )
