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
num_splits = 10

# %%
master_bar, progress_bar = force_console_behavior()

# %%
feature_path = mabe.config.ROOT_PATH / "features_task12.hdf5"

# %%
with h5py.File(feature_path, "r") as hdf:

    def load_all(groupname):
        return list(map(lambda v: v[:].astype(np.float32), hdf[groupname].values()))

    train_X = load_all("train/x")
    train_Y = load_all("train/y")

    train_annotators = np.array(list(map(lambda v: v[()], hdf["train/annotators"].values())))
    num_annotators = len(np.unique(train_annotators))

    test_X = load_all("test/x")
    test_Y = load_all("test/y")
    test_groups = list(map(lambda v: v[()], hdf["test/groups"].values()))

# %%
X = train_X + test_X
Y = train_Y + test_Y

scaler = sklearn.preprocessing.StandardScaler().fit(np.concatenate(X))
X = list(map(lambda x: scaler.transform(x), X))
X_train = list(map(lambda x: scaler.transform(x), train_X))
X_test = list(map(lambda x: scaler.transform(x), test_X))

# %%
sample_lengths = np.array(list(map(len, X)))
p_draw = sample_lengths / np.sum(sample_lengths)

min(sample_lengths), max(sample_lengths)

# %%
for i in range(0, num_splits):
    indices_tr = np.arange(len(X_train))
    indices_te = len(X_train) + np.arange(len(X_test))
    indices = np.arange(len(X))

    # sample until the train split has at least one sample from each annotator
    valid = False
    while not valid:
        train_indices_labeled = np.random.choice(indices_tr, int(0.8 * len(X_train)), replace=False)
        valid = len(np.unique(train_annotators[train_indices_labeled])) == num_annotators

    train_indices_unlabeled = np.random.choice(indices_te, int(0.8 * len(X_test)), replace=False)
    train_indices = np.concatenate((train_indices_labeled, train_indices_unlabeled))
    val_indices_labeled = np.array([i for i in indices_tr if i not in train_indices_labeled])
    val_indices_unlabeled = np.array([i for i in indices_te if i not in train_indices_unlabeled])
    val_indices = np.concatenate((val_indices_labeled, val_indices_unlabeled))

    split = dict(
        indices_tr=indices_tr,
        indices_te=indices_te,
        indices=indices,
        train_indices_labeled=train_indices_labeled,
        train_indices_unlabeled=train_indices_unlabeled,
        train_indices=train_indices,
        val_indices_labeled=val_indices_labeled,
        val_indices_unlabeled=val_indices_unlabeled,
        val_indices=val_indices,
    )

    pickle.dump(split, open(mabe.config.ROOT_PATH / f"split_{i}.pkl", "wb"))
