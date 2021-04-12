import pickle

import h5py
import numba
import numpy as np
import sklearn
import sklearn.preprocessing

import mabe.config


class DataWrapper:
    def __init__(self, feature_path):
        self.vocabulary = ["attack", "investigation", "mount", "other"]

        with h5py.File(feature_path, "r") as hdf:

            def load_all(groupname):
                return list(map(lambda v: v[:].astype(np.float32), hdf[groupname].values()))

            self.train_X = load_all("train/x")
            self.train_Y = load_all("train/y")

            self.test_X = load_all("test/x")
            self.test_Y = load_all("test/y")

            self.test_groups = list(map(lambda v: v[()], hdf["test/groups"].values()))

        self.X = self.train_X + self.test_X
        self.Y = self.train_Y + self.test_Y

        scaler = sklearn.preprocessing.StandardScaler().fit(np.concatenate(self.X))
        self.X = list(map(lambda x: scaler.transform(x), self.X))
        self.X_train = list(map(lambda x: scaler.transform(x), self.train_X))
        self.X_test = list(map(lambda x: scaler.transform(x), self.test_X))

        self.sample_lengths = np.array(list(map(len, self.X)))


class CVSplit:
    def __init__(self, split_idx, data):
        split = pickle.load(open(mabe.config.ROOT_PATH / f"split_{split_idx}.pkl", "rb"))

        self.indices_tr = split["indices_tr"]
        self.indices_te = split["indices_te"]
        self.indices = split["indices"]
        self.train_indices_labeled = split["train_indices_labeled"]
        self.train_indices_unlabeled = split["train_indices_unlabeled"]
        self.train_indices = split["train_indices"]
        self.val_indices_labeled = split["val_indices_labeled"]
        self.val_indices_unlabeled = split["val_indices_unlabeled"]
        self.val_indices = split["val_indices"]

        self.data = data
        self.calculate_draw_probs()

    def calculate_draw_probs(self):
        data = self.data
        sample_lengths = data.sample_lengths

        self.p_draw = sample_lengths / np.sum(sample_lengths)

        self.p_draw_labeled = sample_lengths[self.indices_tr] / np.sum(
            sample_lengths[self.indices_tr]
        )
        self.p_draw_unlabeled = sample_lengths[self.indices_te] / np.sum(
            sample_lengths[self.indices_te]
        )
        self.p_draw_train_labeled = sample_lengths[self.train_indices_labeled] / np.sum(
            sample_lengths[self.train_indices_labeled]
        )
        self.p_draw_train_unlabeled = sample_lengths[self.train_indices_unlabeled] / np.sum(
            sample_lengths[self.train_indices_unlabeled]
        )
        self.p_draw_train = sample_lengths[self.train_indices] / np.sum(
            sample_lengths[self.train_indices]
        )
        self.p_draw_val_labeled = sample_lengths[self.val_indices_labeled] / np.sum(
            sample_lengths[self.val_indices_labeled]
        )
        self.p_draw_val_unlabeled = sample_lengths[self.val_indices_unlabeled] / np.sum(
            sample_lengths[self.val_indices_unlabeled]
        )
        self.p_draw_val = sample_lengths[self.val_indices] / np.sum(
            sample_lengths[self.val_indices]
        )

        _, class_counts = np.unique(np.concatenate(data.train_Y).astype(np.int), return_counts=True)
        self.p_class = class_counts / np.sum(class_counts)

    def get_train_batch(self, batch_size, random_noise=0.0):
        indices_batch = np.concatenate(
            (
                np.random.choice(
                    self.indices_tr, size=int(0.25 * batch_size), p=self.p_draw_labeled
                ),
                np.random.choice(
                    self.indices_te, size=int(0.75 * batch_size), p=self.p_draw_unlabeled
                ),
            )
        )
        X_batch = numba.typed.List()
        augment = lambda x: x + np.random.randn(*x.shape) * random_noise
        if random_noise > 0:
            [X_batch.append(augment(self.data.X[i])) for i in indices_batch]
        else:
            [X_batch.append(self.data.X[i]) for i in indices_batch]
        Y_batch = numba.typed.List()
        [Y_batch.append(self.data.Y[i].astype(int)) for i in indices_batch]

        return X_batch, Y_batch, indices_batch
