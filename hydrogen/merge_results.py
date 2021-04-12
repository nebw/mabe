import collections
import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.decomposition
import sklearn.linear_model
import sklearn.preprocessing
import torch
from fastprogress.fastprogress import force_console_behavior

import mabe
import mabe.config
import mabe.loss
import mabe.model

# %%
master_bar, progress_bar = force_console_behavior()

# %%
feature_path = mabe.config.ROOT_PATH / "features.hdf5"

# %%
submissions = ("submission4.2.npy",)
submissions = list(
    map(lambda p: np.load(mabe.config.ROOT_PATH / p, allow_pickle=True).item(), submissions)
)

# %%
merged_submission = {}

for key in submissions[0].keys():
    preds = np.stack(list(map(lambda s: s[key], submissions)))
    preds = np.argmax(np.mean(preds, axis=0), axis=-1)

    merged_submission[key] = preds

# %%
sample_submission = np.load(
    mabe.config.ROOT_PATH / "sample-submission.npy", allow_pickle=True
).item()


def validate_submission(submission, sample_submission):
    if not isinstance(submission, dict):
        print("Submission should be dict")
        return False

    if not submission.keys() == sample_submission.keys():
        print("Submission keys don't match")
        return False

    for key in submission:
        sv = submission[key]
        ssv = sample_submission[key]
        if not len(sv) == len(ssv):
            print(f"Submission lengths of {key} doesn't match")
            return False

    for key, sv in submission.items():
        if not all(isinstance(x, (np.int32, np.int64, int)) for x in list(sv)):
            print(f"Submission of {key} is not all integers")
            return False

    print("All tests passed")
    return True


if validate_submission(merged_submission, sample_submission):
    np.save(mabe.config.ROOT_PATH / "submission4_only.2.npy", merged_submission)
