import joblib
import numba
import numpy as np
import scipy
import scipy.spatial
from fastprogress.fastprogress import force_console_behavior

import mabe
import mabe.config

master_bar, progress_bar = force_console_behavior()


pose_pca = joblib.load(mabe.config.ROOT_PATH / "pose_pca.joblib")


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


def wall_dist(mid):
    xmax = 1020
    ymax = 570

    wall_dist = np.linalg.norm(
        np.stack(
            (
                np.stack((mid[:, 0], xmax - mid[:, 0])).min(axis=0),
                np.stack((mid[:, 1], ymax - mid[:, 1])).min(axis=0),
            )
        ),
        axis=0,
    )

    return np.tanh(wall_dist / 465)


def get_pdists(trajectory):
    t = trajectory.transpose(0, 1, 3, 2)
    t = t.reshape(t.shape[0], -1, t.shape[-1])[1:]

    X_pdists = np.stack([scipy.spatial.distance.pdist(s) for s in t])
    X_pdists = np.log1p(X_pdists)

    return pose_pca.transform(X_pdists)


def transform_to_feature_vector(
    trajectory, with_abs_pos=False, with_wall_dist=True, with_pdists=True, raw_trajectories=False
):
    if raw_trajectories:
        t = trajectory.transpose(0, 1, 3, 2)
        t = t.reshape(t.shape[0], -1, t.shape[-1])[1:]

        return t, t

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

    features = np.concatenate(
        (m0[1:], m1[1:], velocity, orientation, relative_position_info[1:]), axis=1
    )

    if with_pdists:
        pdists = get_pdists(trajectory)
        features = np.concatenate((features, pdists), axis=1)

    indices = np.arange(0, velocity.shape[0])
    is_beginning = np.tanh(indices / 1500).reshape(-1, 1)
    is_ending = np.tanh((velocity.shape[0] - indices) / 2300).reshape(-1, 1)

    extra_features = np.concatenate(
        (is_beginning, is_ending, wall_dist(mid0)[1:, None], wall_dist(mid1)[1:, None]), axis=1
    )

    return features, extra_features


def get_features_and_labels(sample_sequence, raw_trajectories=False):
    features, extra_features = transform_to_feature_vector(
        sample_sequence["keypoints"], raw_trajectories=raw_trajectories
    )
    if "annotations" in sample_sequence:
        labels = sample_sequence["annotations"]
        labels = labels[1:]
        annotator = sample_sequence["annotator_id"]
    else:
        labels = np.array([-1] * features.shape[0])
        annotator = None

    return features, extra_features, labels, annotator


def load_dataset(path=None, raw_data=None, raw_trajectories=False):
    assert path is not None or raw_data is not None
    if raw_data is None:
        raw_data = np.load(path, allow_pickle=True).item()

    if "vocabulary" in raw_data:
        label_vocabulary = raw_data["vocabulary"]
        print(label_vocabulary)
    raw_data = raw_data["sequences"]

    X = []
    X_extra = []
    Y = []
    groups = []
    annotators = []

    for key, data in raw_data.items():
        x, x_extra, y, annotator = get_features_and_labels(data, raw_trajectories=raw_trajectories)
        X.append(x)
        X_extra.append(x_extra)
        Y.append(y)
        groups.append(key)
        annotators.append(annotator)

    return X, X_extra, Y, groups, annotators


def load_task3_datasets(path, raw_trajectories=False):
    raw_data = np.load(path, allow_pickle=True).item()

    for behavior_key in raw_data.keys():
        raw_data_behavior = raw_data[behavior_key]

        yield behavior_key, load_dataset(
            raw_data=raw_data_behavior, raw_trajectories=raw_trajectories
        )
