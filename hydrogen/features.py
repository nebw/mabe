# %% codecell
import h5py
import joblib
import matplotlib.pyplot as plt
import numba
import numpy as np
import sklearn
import sklearn.decomposition
from fastprogress.fastprogress import force_console_behavior

import mabe
import mabe.config

# %% codecell
master_bar, progress_bar = force_console_behavior()

train_path = mabe.config.ROOT_PATH / "train.npy"
test_path = mabe.config.ROOT_PATH / "test-release.npy"
pca_path = mabe.config.ROOT_PATH / "pose-pca.joblib"
feature_path = mabe.config.ROOT_PATH / "features_pca.hdf5"

# %% codecell
pose_pca = joblib.load(pca_path)

# %% codecell
@numba.njit
def get_mouse_orientation_angle(mouse):
    tail_coords = mouse[:, :, -1]
    neck_coords = mouse[:, :, 3]

    orientation_vector = neck_coords - tail_coords
    body_angle = np.arctan2(orientation_vector[:, 1], orientation_vector[:, 0])

    return tail_coords, neck_coords, orientation_vector, body_angle


@numba.njit
def rotate_broadcast(vector, angle):
    vector = vector.astype(np.float32)
    for t_idx in range(vector.shape[0]):
        c, s = np.cos(-angle[t_idx]), np.sin(-angle[t_idx])
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        vector[t_idx] = np.dot(R, vector[t_idx])
    return vector


def normalize_mouse(mouse):
    tail_coords, neck_coords, orientation_vector, body_angle = get_mouse_orientation_angle(mouse)

    mid_coords = (tail_coords + neck_coords) / 2.0
    mouse -= mid_coords[:, :, None]

    mouse = rotate_broadcast(mouse, body_angle)

    return mouse, mid_coords


def get_distance_angle_to(xy0, xy1):
    vector = xy1 - xy0
    distance = np.linalg.norm(vector, axis=1)[:, None]
    vector = vector / distance
    return vector, distance


def get_distance_angle_between_mice(m0, m1):
    neck0 = m0[:, :, 3]
    neck1 = m1[:, :, 3]
    butt1 = m1[:, :, -1]

    tail_coords, neck_coords, orientation_vector, body_angle = get_mouse_orientation_angle(m0)

    v0, d0 = get_distance_angle_to(neck0, neck1)
    v0 = rotate_broadcast(v0, body_angle)
    v1, d1 = get_distance_angle_to(neck0, butt1)
    v1 = rotate_broadcast(v1, body_angle)

    return np.concatenate((v0, d0, v1, d1), axis=1)


def get_movement_velocity_orienation(mouse):
    tail_coords, neck_coords, orientation_vector, body_angle = get_mouse_orientation_angle(mouse)

    mean_mouse = np.mean(mouse, axis=2)
    mean_motion = np.diff(mean_mouse, axis=0)

    velocity = np.einsum("bc,bc->b", orientation_vector[1:], mean_motion)[:, None]

    angle_diff = np.diff(body_angle, axis=0)
    orientation_change = np.stack((np.cos(angle_diff), np.sin(angle_diff)), axis=1)

    return velocity, orientation_change


def transform_to_feature_vector(trajectory, with_abs_pos=False, with_pca=True):
    m0 = trajectory[:, 0, :, :].copy()
    m1 = trajectory[:, 1, :, :].copy()
    velocity, orientation = get_movement_velocity_orienation(m0)
    relative_position_info = get_distance_angle_between_mice(m0, m1)
    m0, mid0 = normalize_mouse(m0)
    m1, mid1 = normalize_mouse(m1)

    if with_abs_pos:
        m0 = np.concatenate((m0, mid0[:, :, None]), axis=-1)
        m1 = np.concatenate((m1, mid1[:, :, None]), axis=-1)

    m0 = m0.reshape(-1, m0.shape[1] * m0.shape[2])
    m1 = m1.reshape(-1, m1.shape[1] * m1.shape[2])

    if with_pca:
        assert not with_abs_pos

        m0 = pose_pca.transform(m0)
        m1 = pose_pca.transform(m1)

    features = np.concatenate(
        (m0[1:], m1[1:], velocity, orientation, relative_position_info[1:]), axis=1
    )
    return features


def get_features_and_labels(sample_sequence):

    features = transform_to_feature_vector(sample_sequence["keypoints"])
    if "annotations" in sample_sequence:
        labels = sample_sequence["annotations"]
        labels = labels[1:]
    else:
        labels = np.array([-1] * features.shape[0])

    return features, labels


def load_dataset(path):
    raw_data = np.load(path, allow_pickle=True).item()
    if "vocabulary" in raw_data:
        label_vocabulary = raw_data["vocabulary"]
        print(label_vocabulary)
    raw_data = raw_data["sequences"]

    X = []
    Y = []
    groups = []

    for key, data in progress_bar(raw_data.items()):
        x, y = get_features_and_labels(data)
        X.append(x)
        Y.append(y)
        groups.append(key)

    return X, Y, groups


# %% codecell
X, Y, groups = load_dataset(train_path)

# %% codecell
X_test, Y_test, groups_test = load_dataset(test_path)

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
    store("train/y", Y)
    store("train/groups", groups)

    store("test/x", X_test)
    store("test/y", Y_test)
    store("test/groups", groups_test)
