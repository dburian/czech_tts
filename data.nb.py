# %%
# %load_ext autoreload
# %autoreload 1

# %% [markdown]
# ## Duration of whole dataset

# %%
import pandas as pd

# %%
data = pd.read_csv(
    "/home/dburian/downloads/cv-corpus-19.0-2024-09-13/cs/train.tsv", sep="\t"
)

# %%
data.shape

# %%
data.columns

# %%
import torchaudio


# %%
def _duration(mp3_path: str) -> float:
    wav, sr = torchaudio.load(
        f"/home/dburian/downloads/cv-corpus-19.0-2024-09-13/cs/clips/{mp3_path}"
    )
    return wav.shape[1] / sr


# %%
data["duration"] = data["path"].apply(_duration)

# %%
data["duration"].sum() / 60**2

# %%
durations = data.groupby("client_id")["duration"].sum() / 60**2

# %%
durations

# %%
data["sentence_id"].unique().shape

# %%
data["sentence"].iloc[4]

# %%
data["path"].iloc[2]

# %% [markdown]
# ## Testing the loader

# %%
from czech_tts.loader import Loader
from tqdm.auto import tqdm

# %%
loader = Loader("./cv-corpus-19.0-2024-09-13/", "train")
# %%
len(loader)

# %%
rec = next(iter(loader))

# %%
rec

# %%
loader.dump_mfa("./cv-corpus-19.0-2024-09-13/cs/mfa")

# %%
print("done")
