import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from bayes_opt import BayesianOptimization, UtilityFunction
from matplotlib import gridspec

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

train["vocabulary"]

# %%
sample.keys()

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
                lambda s: np.stack(
                    (
                        np.arange(len(s["annotations"])),
                        np.arange(len(s["annotations"]))[::-1],
                        s["annotations"],
                    )
                ),
                train["sequences"].values(),
            )
        ),
        axis=1,
    ).T,
    columns=["timestep", "timestep_reverse", "annotation"],
)

# %%
annotation_df.groupby("timestep").apply(lambda ts: (ts.annotation == 0).mean()).plot()

# %%
annotation_df.groupby("timestep_reverse").apply(lambda ts: (ts.annotation == 0).mean()).plot()

# %%
annotation_df.groupby("timestep").apply(lambda ts: (ts.annotation == 1).mean()).plot()

# %%
annotation_df.groupby("timestep_reverse").apply(lambda ts: (ts.annotation == 1).mean()).plot()

# %%
annotation_df.groupby("timestep").apply(lambda ts: (ts.annotation == 2).mean()).plot()

# %%
annotation_df.groupby("timestep").apply(lambda ts: (ts.annotation == 3).mean()).plot()

# %%
keypoints = list(map(lambda s: s["keypoints"], train["sequences"].values()))
groups = np.concatenate([np.ones(len(keypoints[i])) * i for i in range(len(keypoints))]).astype(
    np.int
)

# %%
linear = sklearn.linear_model.LogisticRegression(
    multi_class="multinomial", max_iter=1000, class_weight="balanced"
)

# %%
def evaluate_f1(c):
    l = linear.fit(
        np.tanh(annotation_df[["timestep"]] / c),
        annotation_df["annotation"],
    )
    return sklearn.metrics.f1_score(
        annotation_df["annotation"],
        l.predict(
            np.tanh(annotation_df[["timestep"]] / c),
        ),
        average="macro",
    )


# %%
pbounds = {"c": (1, 5000)}

optimizer = BayesianOptimization(
    f=evaluate_f1,
    pbounds=pbounds,
    random_state=42,
)

optimizer.maximize(
    init_points=5,
    n_iter=50,
)

print(optimizer.max)

# %%
def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer, x):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        f"Gaussian Process and Utility Function After {steps} Steps", fontdict={"size": 30}
    )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = np.array([[res["params"]["c"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x_obs.flatten(), y_obs, "D", markersize=8, label="Observations", color="r")
    axis.plot(x, mu, "--", color="k", label="Prediction")

    axis.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=0.6,
        fc="c",
        ec="None",
        label="95% confidence interval",
    )

    axis.set_ylim((None, None))
    axis.set_ylabel("f(x)", fontdict={"size": 20})
    axis.set_xlabel("x", fontdict={"size": 20})

    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label="Utility Function", color="purple")
    acq.plot(
        x[np.argmax(utility)],
        np.max(utility),
        "*",
        markersize=15,
        label="Next Best Guess",
        markerfacecolor="gold",
        markeredgecolor="k",
        markeredgewidth=1,
    )
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel("Utility", fontdict={"size": 20})
    acq.set_xlabel("x", fontdict={"size": 20})

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.0)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.0)


# %%
plot_gp(optimizer, np.linspace(*pbounds["c"], num=1000)[:, None])

# %%
def evaluate_f1_reverse(c):
    l = linear.fit(
        np.tanh(annotation_df[["timestep_reverse"]] / c),
        annotation_df["annotation"],
    )
    return sklearn.metrics.f1_score(
        annotation_df["annotation"],
        l.predict(
            np.tanh(annotation_df[["timestep_reverse"]] / c),
        ),
        average="macro",
    )


# %%
pbounds = {"c": (1, 10000)}

optimizer = BayesianOptimization(
    f=evaluate_f1_reverse,
    pbounds=pbounds,
    random_state=42,
)

optimizer.maximize(
    init_points=5,
    n_iter=50,
)

print(optimizer.max)

# %%
plot_gp(optimizer, np.linspace(*pbounds["c"], num=1000)[:, None])

# %%
def evaluate_f1_combined(c0, c1):
    features = np.tanh(
        np.concatenate(
            (annotation_df[["timestep"]] / c0, annotation_df[["timestep_reverse"]] / c1), axis=-1
        )
    )
    l = linear.fit(
        features,
        annotation_df["annotation"],
    )
    return sklearn.metrics.f1_score(
        annotation_df["annotation"],
        l.predict(features),
        average="macro",
    )


# %%
pbounds = {"c0": (1, 10000), "c1": (1, 10000)}

optimizer = BayesianOptimization(
    f=evaluate_f1_combined,
    pbounds=pbounds,
    random_state=42,
)

optimizer.maximize(
    init_points=5,
    n_iter=500,
)

print(optimizer.max)

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

# %%
c0 = 1485.151664444058
c1 = 2308.0343145614083


features = np.tanh(
    np.concatenate(
        (annotation_df[["timestep"]] / c0, annotation_df[["timestep_reverse"]] / c1), axis=-1
    )
)
wall_dist = np.linalg.norm(
    np.stack(
        (
            np.stack((annotation_df.x, annotation_df.x.max() - annotation_df.x)).min(axis=0),
            np.stack((annotation_df.y, annotation_df.y.max() - annotation_df.y)).min(axis=0),
        )
    ),
    axis=0,
)

# %%
def evaluate_f1_wdist(c):
    features = np.tanh(wall_dist / c)[:, None]
    l = linear.fit(
        features,
        annotation_df["annotation"],
    )
    return sklearn.metrics.f1_score(
        annotation_df["annotation"],
        l.predict(features),
        average="macro",
    )


# %%
pbounds = {"c": (1, 600)}

optimizer = BayesianOptimization(
    f=evaluate_f1_wdist,
    pbounds=pbounds,
    random_state=42,
)

optimizer.maximize(
    init_points=5,
    n_iter=50,
)

print(optimizer.max)

# %%
plot_gp(optimizer, np.linspace(*pbounds["c"], num=1000)[:, None])
