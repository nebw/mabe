import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mabe
import mabe.config

# %%
train = np.load(mabe.config.ROOT_PATH / "train.npy", allow_pickle=True).item()
test = np.load(mabe.config.ROOT_PATH / "test-release.npy", allow_pickle=True).item()
sample_submission = np.load(
    mabe.config.ROOT_PATH / "sample-submission.npy", allow_pickle=True
).item()

# %%
sample_key = list(train["sequences"].keys())[4]
sample = train["sequences"][sample_key]

# Dimensions: (# frames) x (mouse ID) x (x, y coordinate) x (body part).
# Units: pixels; coordinates are relative to the entire image.
# Original image dimensions are 1024 x 570.

# %%
sample.keys()

# %%
plt.scatter(np.arange(len(sample["annotations"])), sample["annotations"])

# %%
sample["keypoints"].shape

# %%
plt.scatter(*sample["keypoints"][-1000][0], c="blue")
plt.scatter(*sample["keypoints"][-1000][1], c="red")

# %%
groups = train["sequences"].keys()
annotation_df = pd.DataFrame(
    np.concatenate(
        list(
            map(
                lambda s: np.stack((np.arange(len(s["annotations"])), s["annotations"])),
                train["sequences"].values(),
            )
        ),
        axis=1,
    ).T,
    columns=["timestep", "annotation"],
)

# %%
annotation_df.groupby("timestep").apply(lambda ts: (ts.annotation == 0).mean()).plot()

# %%
annotation_df.groupby("timestep").apply(lambda ts: (ts.annotation == 1).mean()).plot()

# %%
annotation_df.groupby("timestep").apply(lambda ts: (ts.annotation == 2).mean()).plot()

# %%
annotation_df.groupby("timestep").apply(lambda ts: (ts.annotation == 3).mean()).plot()

# %%
import sklearn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

# %%
keypoints = list(map(lambda s: s["keypoints"], train["sequences"].values()))
groups = np.concatenate([np.ones(len(keypoints[i])) * i for i in range(len(keypoints))]).astype(
    np.int
)

# %%
linear = sklearn.linear_model.LogisticRegression(
    multi_class="multinomial", max_iter=1000, class_weight="balanced"
)
scores = sklearn.model_selection.cross_validate(
    linear,
    np.tanh(annotation_df[["timestep"]] / 1000),
    annotation_df[["annotation"]],
    n_jobs=8,
    cv=sklearn.model_selection.GroupShuffleSplit(32),
    groups=groups,
    scoring=dict(
        f1=sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average="macro"),
        precision=sklearn.metrics.make_scorer(sklearn.metrics.precision_score, average="macro"),
    ),
)
np.median(scores["test_f1"]), np.median(scores["test_precision"])

# %%
annotation_df["x"] = np.concatenate(list(map(lambda k: k[:, 0, :, 0], keypoints)))[:, 0]
annotation_df["y"] = np.concatenate(list(map(lambda k: k[:, 0, :, 0], keypoints)))[:, 1]

annotation_df["qx"] = pd.cut(annotation_df.x, 20, labels=False)
annotation_df["qy"] = pd.cut(annotation_df.y, 20, labels=False)

# %%
plt.imshow(
    annotation_df.pivot_table(
        index="qx", columns="qy", values="annotation", aggfunc=lambda v: (v == 0).mean()
    )
)
plt.title("Attack")
plt.show()

# %%
plt.imshow(
    annotation_df.pivot_table(
        index="qx", columns="qy", values="annotation", aggfunc=lambda v: (v == 1).mean()
    )
)
plt.title("Investigation")
plt.show()

# %%
plt.imshow(
    annotation_df.pivot_table(
        index="qx", columns="qy", values="annotation", aggfunc=lambda v: (v == 2).mean()
    )
)
plt.title("Mount")
plt.show()

# %%
plt.imshow(
    annotation_df.pivot_table(
        index="qx", columns="qy", values="annotation", aggfunc=lambda v: (v == 3).mean()
    )
)
plt.title("Other")
plt.colorbar()
plt.show()
