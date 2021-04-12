import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.special
import torch
from fastprogress.fastprogress import force_console_behavior

import mabe
import mabe.config
import mabe.loss
import mabe.model
import mabe.training

# %%
master_bar, progress_bar = force_console_behavior()

# %%
results_path = (
    mabe.config.ROOT_PATH / "training_results_2021-04-10 19:36:30.018611_baseline_madgrad_0.834.pt"
)
results = torch.load(results_path)

# %%
keys = results[0].test_logits.keys()

merged_submission = {}
for key in keys:
    preds = np.stack(list(map(lambda s: s.test_logits[key], results)))
    preds = np.argmax(scipy.special.softmax(preds, -1).mean(axis=0), axis=-1)

    merged_submission[key] = preds


plt.plot(preds)


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
    np.save(mabe.config.ROOT_PATH / "submission5.npy", merged_submission)
