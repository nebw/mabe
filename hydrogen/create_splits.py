import pickle

import h5py
import numpy as np
import sklearn
import sklearn.decomposition
import sklearn.linear_model
import sklearn.preprocessing
from fastprogress.fastprogress import force_console_behavior

import mabe
import mabe.config
import mabe.loss
import mabe.model
import mabe.ringbuffer

# %%
num_splits = 32

# %%
master_bar, progress_bar = force_console_behavior()

# %%
feature_path = mabe.config.ROOT_PATH / "features_task123.hdf5"

# %%
with h5py.File(feature_path, "r") as hdf:

    def load_all(groupname):
        return list(map(lambda v: v[:].astype(np.float32), hdf[groupname].values()))

    X_labeled = load_all("train/x")
    Y_labeled = load_all("train/y")

    annotators_labeled = np.array(list(map(lambda v: v[()], hdf["train/annotators"].values())))
    num_annotators = len(np.unique(annotators_labeled))

    clf_tasks_labeled = np.array(list(map(lambda v: v[()], hdf["train/clf_tasks"].values())))
    num_clf_tasks = len(np.unique(clf_tasks_labeled))

    X_unlabeled = load_all("test/x")
    Y_unlabeled = load_all("test/y")
    groups_unlabeled = list(map(lambda v: v[()], hdf["test/groups"].values()))

# %%
X = X_labeled + X_unlabeled
Y = Y_labeled + Y_unlabeled

scaler = sklearn.preprocessing.StandardScaler().fit(np.concatenate(X))
X = list(map(lambda x: scaler.transform(x), X))
X_labeled = list(map(lambda x: scaler.transform(x), X_labeled))
X_unlabeled = list(map(lambda x: scaler.transform(x), X_unlabeled))

# %%
sample_lengths = np.array(list(map(len, X)))
p_draw = sample_lengths / np.sum(sample_lengths)

len(X), len(X_labeled), min(sample_lengths), max(sample_lengths)

# %%
for i in range(0, num_splits):
    indices_labeled = np.arange(len(X_labeled))
    indices_unlabeled = len(X_labeled) + np.arange(len(X_unlabeled))
    indices = np.arange(len(X))

    # sample until the train split has at least one sample from each annotator
    valid = False
    while not valid:
        train_indices_labeled = np.random.choice(
            indices_labeled, int(0.85 * len(X_labeled)), replace=False
        )
        val_indices_labeled = np.array(
            [i for i in indices_labeled if i not in train_indices_labeled]
        )

        valid = len(np.unique(annotators_labeled[train_indices_labeled])) == num_annotators
        valid &= len(np.unique(clf_tasks_labeled[train_indices_labeled])) == num_clf_tasks
        valid &= len(np.unique(clf_tasks_labeled[val_indices_labeled])) >= (
            num_clf_tasks - 1
        )  # one task with only one trajectory

    train_indices_unlabeled = np.random.choice(
        indices_unlabeled, int(0.85 * len(X_unlabeled)), replace=False
    )
    train_indices = np.concatenate((train_indices_labeled, train_indices_unlabeled))
    val_indices_unlabeled = np.array(
        [i for i in indices_unlabeled if i not in train_indices_unlabeled]
    )
    val_indices = np.concatenate((val_indices_labeled, val_indices_unlabeled))

    split = dict(
        indices_labeled=indices_labeled,
        indices_unlabeled=indices_unlabeled,
        indices=indices,
        train_indices_labeled=train_indices_labeled,
        train_indices_unlabeled=train_indices_unlabeled,
        train_indices=train_indices,
        val_indices_labeled=val_indices_labeled,
        val_indices_unlabeled=val_indices_unlabeled,
        val_indices=val_indices,
    )

    pickle.dump(split, open(mabe.config.ROOT_PATH / f"split_{i}.pkl", "wb"))
