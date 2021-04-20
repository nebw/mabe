# %% codecell
import h5py
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
pca_path = mabe.config.ROOT_PATH / "pose-pca.joblib"
feature_path = mabe.config.ROOT_PATH / "features_task123.hdf5"

# %% codecell
X, X_extra, Y, groups, annotators = mabe.features.load_dataset(train_path)
Xt2, Xt2_extra, Yt2, groupst2, annotatorst2 = mabe.features.load_dataset(train_task2_path)

X += Xt2
X_extra += Xt2_extra
Y += Yt2
groups += groupst2
annotators += annotatorst2
clf_tasks = [0] * len(X)

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
"""
X_m0 = np.concatenate([x[:, :14] for x in X])
X_m1 = np.concatenate([x[:, 14:14+14] for x in X])
X_test_m0 = np.concatenate([x[:, :14] for x in X_test])
X_test_m1 = np.concatenate([x[:, 14:14+14] for x in X_test])
mouse_poses = np.concatenate((X_m0, X_m1, X_test_m0, X_test_m1))

pca = sklearn.decomposition.PCA(n_components=11)
pca.fit(mouse_poses)

joblib.dump(pca, pca_path)
"""

# %% codecell
len(X), len(Y), len(groups)

# %% codecell
X[0].shape, X[1].shape, Y[0].shape

# %% codecell
len(X_test), len(Y_test), len(groups_test)

# %% codecell
X_test[0].shape, X_test[1].shape, Y_test[0].shape

# %% codecell
with h5py.File(feature_path, "w") as hdf:

    def store(groupname, values):
        grp = hdf.create_group(groupname)
        for idx, v in enumerate(values):
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
    store("test/groups", groups_test)
