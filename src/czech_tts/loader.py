import os
import json
from typing import Iterable, TypedDict
import pandas as pd
import numpy as np
from torchaudio import load
import torchaudio
from tqdm.auto import tqdm
import torch


class Recording(TypedDict):
    id: int
    waveform: np.ndarray
    speaker_id: str
    text: str


class Loader:
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

            speaker_root = os.path.join(mfa_root_path, speaker_id)
            os.makedirs(speaker_root, exist_ok=True)
            torchaudio.save(
                os.path.join(speaker_root, f"{idx}.wav"),
                torch.from_numpy(waveform).unsqueeze(0),
                self.sr,
                channels_first=True,
                format="wav",
            )
            with open(os.path.join(speaker_root, f"{idx}.txt"), mode="w") as txt_file:
                print(text, file=txt_file, end="")


# class Recording(TypedDict):
#     id: int
#     phonemes: list[str]
#     durations: list[float]
#     waveform: np.ndarray


# class Loader:
#     def __init__(self, root_path: str, sampling_rate: int = 16000) -> None:
#         self.root_path = root_path
#         self._len = None
#         self.sr = sampling_rate

#     @property
#     def speech_path(self) -> str:
#         return os.path.join(self.root_path, "speech_48kHz")

#     @property
#     def segment_path(self) -> str:
#         return os.path.join(self.root_path, "segmentation")

#     @property
#     def duration_path(self) -> str:
#         return os.path.join(self.root_path, "duration")

#     @property
#     def pitch_path(self) -> str:
#         return os.path.join(self.root_path, "pitch")

#     def __len__(self) -> int:
#         if self._len is None:
#             self._len = len(os.listdir(self.speech_path))

#         return self._len

#     def __iter__(self) -> Iterable[Recording]:
#         for i in range(len(self)):
#             phonemes, durations = self._load_segmentation(i)
#             yield {
#                 "id": i,
#                 "phonemes": phonemes,
#                 "durations": durations,
#                 "waveform": self._load_waveform(i),
#             }

#     def _load_waveform(self, id: int) -> np.ndarray:
#         wav, sr = load(
#             os.path.join(self.speech_path, f"cz-ham-{id:0>5}.wav"),
#             format="wav",
#             backend="ffmpeg",
#         )

#         if sr != self.sr:
#             wav = torchaudio.functional.resample(wav, sr, self.sr)

#         return wav.numpy()[0]

#     def _load_segmentation(self, id: int) -> tuple[list[str], list[float]]:
#         with open(os.path.join(self.segment_path, f"cz-ham-{id:0>5}.json")) as infile:
#             segmentation = json.load(infile)

#             phonemes = []
#             durations = []
#             for seg in segmentation:
#                 if seg["type"] != "phone":
#                     continue

#                 phoneme = seg["text"]
#                 if phoneme == "#":
#                     phoneme = "$"
#                 phonemes.append(phoneme)
#                 durations.append(float(seg["endTime"]) - float(seg["begTime"]))

#             return phonemes, durations
