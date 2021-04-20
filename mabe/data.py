import dataclasses
import pickle

import h5py
import numba
import numpy as np
import sklearn
import sklearn.preprocessing

import mabe.config


@dataclasses.dataclass
class TrainingBatch:
    X: numba.typed.List  # [np.array]
    X_extra: numba.typed.List  # [np.array]
    Y: numba.typed.List  # [np.array]
    indices: np.array
    annotators: numba.typed.List  # [np.array]
    clf_tasks: np.array


class DataWrapper:
    def __init__(self, feature_path):
        self.vocabulary = ["attack", "investigation", "mount", "other"]

        with h5py.File(feature_path, "r") as hdf:

            def load_all(groupname):
                return list(map(lambda v: v[:].astype(np.float32), hdf[groupname].values()))

            self.X_labeled = load_all("train/x")
            self.Y_labeled = load_all("train/y")

            self.annotators_labeled = list(map(lambda v: v[()], hdf["train/annotators"].values()))
            self.clf_tasks_labeled = np.array(
                list(map(lambda v: int(v[()]), hdf["train/clf_tasks"].values()))
            )

            self.X_unlabeled = load_all("test/x")
            self.Y_unlabeled = load_all("test/y")

            self.annotators_unlabeled = [-1] * len(self.X_unlabeled)
            self.clf_tasks_unlabeled = np.array([-1] * len(self.X_unlabeled))

            try:
                self.X_labeled_extra = load_all("train/x_extra")
                self.X_unlabeled_extra = load_all("test/x_extra")
            except KeyError:
                self.X_labeled_extra = None
                self.X_unlabeled_extra = None

            self.groups_unlabeled = list(map(lambda v: v[()], hdf["test/groups"].values()))

        self.X = self.X_labeled + self.X_unlabeled
        self.Y = self.Y_labeled + self.Y_unlabeled
        self.annotators = self.annotators_labeled + self.annotators_unlabeled
        self.num_annotators = len(np.unique(self.annotators_labeled))
        self.clf_tasks = np.concatenate((self.clf_tasks_labeled, self.clf_tasks_unlabeled))
        self.num_clf_tasks = len(np.unique(self.clf_tasks))

        if self.X_labeled_extra is not None:
            self.X_extra = self.X_labeled_extra + self.X_unlabeled_extra
            self.num_extra_features = self.X_extra[0].shape[-1]
        else:
            self.X_extra = None
            self.num_extra_features = 0

        scaler = sklearn.preprocessing.StandardScaler().fit(np.concatenate(self.X))
        self.X = list(map(lambda x: scaler.transform(x), self.X))
        self.X_labeled = list(map(lambda x: scaler.transform(x), self.X_labeled))
        self.X_unlabeled = list(map(lambda x: scaler.transform(x), self.X_unlabeled))

        self.sample_lengths = np.array(list(map(len, self.X)))


class CVSplit:
    def __init__(self, split_idx, data):
        split = pickle.load(open(mabe.config.ROOT_PATH / f"split_{split_idx}.pkl", "rb"))

        self.indices_labeled = split["indices_labeled"]
        self.indices_unlabeled = split["indices_unlabeled"]
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

        self.p_draw_labeled = sample_lengths[self.indices_labeled] / np.sum(
            sample_lengths[self.indices_labeled]
        )
        self.p_draw_unlabeled = sample_lengths[self.indices_unlabeled] / np.sum(
            sample_lengths[self.indices_unlabeled]
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

    def get_train_batch(self, batch_size, random_noise=0.0, extra_features=False):
        def random_task_train_index(task):
            task_train_indices = self.train_indices_labeled[
                self.data.clf_tasks[self.train_indices_labeled] == task
            ]
            task_p_draw = self.data.sample_lengths[task_train_indices].astype(np.float)
            task_p_draw /= np.sum(task_p_draw)
            return np.array([np.random.choice(task_train_indices, p=task_p_draw)])

        # at least one sample per task
        indices_batch = np.concatenate(
            (
                np.random.choice(
                    self.train_indices_labeled,
                    size=int(0.25 * batch_size),
                    p=self.p_draw_train_labeled,
                ),
                *[
                    random_task_train_index(task)
                    for task in range(0, self.data.clf_tasks.max() + 1)
                ],
                np.random.choice(
                    self.indices_unlabeled,
                    size=int(0.75 * batch_size - self.data.clf_tasks.max() - 1),
                    p=self.p_draw_unlabeled,
                ),
            )
        )

        assert np.all([i not in self.val_indices_labeled for i in indices_batch])

        X_batch = numba.typed.List()
        augment = lambda x: x + np.random.randn(*x.shape) * random_noise
        if random_noise > 0:
            [X_batch.append(augment(self.data.X[i])) for i in indices_batch]
        else:
            [X_batch.append(self.data.X[i]) for i in indices_batch]
        Y_batch = numba.typed.List()
        [Y_batch.append(self.data.Y[i].astype(int)) for i in indices_batch]

        annotators_batch = numba.typed.List()
        [
            annotators_batch.append(np.array([self.data.annotators[i]]).repeat(len(y)))
            for i, y in zip(indices_batch, Y_batch)
        ]

        clf_tasks_batch = self.data.clf_tasks[indices_batch]

        X_extra_batch = None
        if extra_features:
            X_extra_batch = numba.typed.List()
            [X_extra_batch.append(self.data.X_extra[i].astype(int)) for i in indices_batch]

        return TrainingBatch(
            X_batch, X_extra_batch, Y_batch, indices_batch, annotators_batch, clf_tasks_batch
        )
