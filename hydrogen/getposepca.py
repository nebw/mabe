import fastprogress
import joblib
import numpy as np
import scipy.spatial
import scipy.spatial.distance
import scipy.special
import sklearn
import sklearn.decomposition

import mabe
import mabe.config
import mabe.features

# %% codecell
master_bar, progress_bar = fastprogress.fastprogress.force_console_behavior()

train_path = mabe.config.ROOT_PATH / "train.npy"
train_task2_path = mabe.config.ROOT_PATH / "train_task2.npy"
train_task3_path = mabe.config.ROOT_PATH / "train_task3.npy"
test_path = mabe.config.ROOT_PATH / "test-release.npy"
pca_path = mabe.config.ROOT_PATH / "pose-pca.joblib"

# %%
test = np.load(mabe.config.ROOT_PATH / "test-release.npy", allow_pickle=True).item()

# %% codecell
X, X_extra, Y, groups, annotators = mabe.features.load_dataset(train_path, raw_trajectories=True)

# %%
Xt2, Xt2_extra, Yt2, groupst2, annotatorst2 = mabe.features.load_dataset(
    train_task2_path, raw_trajectories=True
)

X += Xt2
X_extra += Xt2_extra
Y += Yt2
groups += groupst2
annotators += annotatorst2
clf_tasks = [0] * len(X)

for behavior, data in mabe.features.load_task3_datasets(train_task3_path, raw_trajectories=True):
    clf_task = int(behavior[-1]) + 1
    print(behavior, clf_task)

    X += data[0]
    X_extra += data[1]
    Y += data[2]
    groups += data[3]
    annotators += data[4]
    clf_tasks += [clf_task] * len(data[0])

# %% codecell
X_test, X_extra_test, Y_test, groups_test, _ = mabe.features.load_dataset(
    test_path, raw_trajectories=True
)

# %%
X += X_test
Y += Y_test
groups += groups_test

# %%
X_pdists = [np.stack([scipy.spatial.distance.pdist(s) for s in x]) for x in X]
X_pdists = np.concatenate(X_pdists)
X_pdists = np.log1p(X_pdists)

# %%
pca = sklearn.decomposition.PCA(0.95)
pca.fit(X_pdists)

np.set_printoptions(suppress=True)
np.cumsum(pca.explained_variance_ratio_)

joblib.dump(pca, mabe.config.ROOT_PATH / "pose_pca.joblib")
