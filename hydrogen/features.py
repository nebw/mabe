# %% codecell
import io

import h5py
import numpy as np
import scipy.spatial
import scipy.spatial.distance
import scipy.special
import torch
from fastprogress.fastprogress import force_console_behavior

import mabe
import mabe.config
import mabe.features

# %% codecell
master_bar, progress_bar = force_console_behavior()

train_path = mabe.config.ROOT_PATH / "train.npy"
train_task2_path = mabe.config.ROOT_PATH / "train_task2.npy"
train_task3_path = mabe.config.ROOT_PATH / "train_task3.npy"
test_path = mabe.config.ROOT_PATH / "test-release.npy"
feature_path = mabe.config.ROOT_PATH / "features_task123_final_pca.hdf5"

# %%
test = np.load(mabe.config.ROOT_PATH / "test-release.npy", allow_pickle=True).item()

# %% codecell
X, X_extra, Y, groups, annotators = mabe.features.load_dataset(train_path)
clf_tasks = [0] * len(X)

Xt2, Xt2_extra, Yt2, groupst2, annotatorst2 = mabe.features.load_dataset(train_task2_path)

X += Xt2
X_extra += Xt2_extra
Y += Yt2
groups += groupst2
annotators += annotatorst2
clf_tasks = [0] * len(X)

# %%
for behavior, data in mabe.features.load_task3_datasets(train_task3_path):
    clf_task = int(behavior[-1]) + 1

    X += data[0]
    X_extra += data[1]
    Y += data[2]
    groups += data[3]
    annotators += data[4]
    clf_tasks += [clf_task] * len(data[0])

# %% codecell
X_test, X_extra_test, Y_test, groups_test, _ = mabe.features.load_dataset(test_path)

# %%
results_files = (
    "training_results_2021-05-03 07:12:21.568593_teacher_ensemble2_0to5_0.844.pt",
    "training_results_2021-05-03 07:30:45.111496_teacher_ensemble2_10to15_0.829.pt",
    "training_results_2021-05-03 19:30:01.286213_teacher_ensemble2_10to15_0.827.pt",
    "training_results_2021-05-03 19:50:54.641450_teacher_ensemble2_15to20_0.814.pt",
    "training_results_2021-05-03 07:20:44.078873_teacher_ensemble2_20to25_0.832.pt",
    "training_results_2021-05-03 19:50:16.149305_teacher_ensemble2_25to30_0.807.pt",
)

# %%
def get_annotator_logits(results, annotator_id="keep"):
    if annotator_id == "keep":
        return results.test_logits
    else:
        assert isinstance(annotator_id, int)
        test_logits = {}
        for key, value in results.test_logits.items():
            test_logits[key] = {annotator_id: value[annotator_id]}
        return test_logits


results = []
for filename in progress_bar(results_files):
    results += list(map(get_annotator_logits, torch.load(mabe.config.ROOT_PATH / filename)))

len(results)

# %%
all_annotators = list({s["annotator_id"] for s in test["sequences"].values()})
# all_annotators = [0]

# %%
Y_test_dark_annotators = []
for key, sequence in progress_bar(test["sequences"].items()):
    all_probs = []
    for annotator_id in all_annotators:
        preds = np.stack(
            list(
                map(
                    lambda s: np.load(io.BytesIO(s[key][annotator_id]))["arr_0"],
                    results,
                )
            )
        )
        all_probs.append(scipy.special.softmax(preds, -1).mean(axis=0))
        assert np.all(np.isfinite(all_probs[-1]))

    all_probs = np.stack(all_probs).transpose(1, 0, 2)
    Y_test_dark_annotators.append(all_probs)

# %%
sample_submission = np.load(
    mabe.config.ROOT_PATH / "sample-submission-task3.npy", allow_pickle=True
).item()

# %%
results = []
for filename in progress_bar(results_files):
    results += list(
        map(lambda r: r.task3_test_logits, torch.load(mabe.config.ROOT_PATH / filename))
    )

len(results)

# %%
Y_test_dark_behaviors = []
for key, sequence in progress_bar(test["sequences"].items()):
    all_probs = []
    for behavior_key in sorted(sample_submission.keys()):
        behavior_idx = int(behavior_key[-1])

        preds = np.stack(
            list(
                map(
                    lambda s: np.load(io.BytesIO(s[key][behavior_idx + 1]))["arr_0"],
                    results,
                )
            )
        )
        all_probs.append(scipy.special.softmax(preds, -1).mean(axis=0))
        assert np.all(np.isfinite(all_probs[-1]))

    all_probs = np.stack(all_probs).transpose(1, 0, 2)
    Y_test_dark_behaviors.append(all_probs)

# %% codecell
with h5py.File(feature_path, "w") as hdf:

    def store(groupname, values):
        grp = hdf.create_group(groupname)
        for idx, v in enumerate(values):
            # grp.create_dataset(f"{idx:04d}", data=v, compression='lzf')
            grp[f"{idx:04d}"] = v

    store("train/x", X)
    store("train/x_extra", X_extra)
    store("train/y", Y)
    store("train/groups", groups)
    store("train/annotators", annotators)
    store("train/clf_tasks", clf_tasks)

    store("test/x", X_test)
    store("test/x_extra", X_extra_test)
    store("test/y", Y_test)
    store("test/y_dark_annotators", Y_test_dark_annotators)
    store("test/y_dark_behaviors", Y_test_dark_behaviors)
    store("test/groups", groups_test)
