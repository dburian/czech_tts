import os
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
