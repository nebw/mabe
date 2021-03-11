import itertools

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
from fastprogress.fastprogress import force_console_behavior

import mabe
import mabe.config

# %%
master_bar, progress_bar = force_console_behavior()

# %%
feature_path = mabe.config.ROOT_PATH / "features.hdf5"

# %%
# train = np.load(mabe.config.ROOT_PATH / "train.npy", allow_pickle=True).item()
# test = np.load(mabe.config.ROOT_PATH / "test-release.npy", allow_pickle=True).item()
# sample_submission = np.load(
#    mabe.config.ROOT_PATH / "sample-submission.npy", allow_pickle=True
# ).item()

# %%
with h5py.File(feature_path, "r") as hdf:

    def load_all(groupname):
        return list(map(lambda v: v[:], hdf[groupname].values()))

    train_X = load_all("train/x")
    train_Y = load_all("train/y")

    test_X = load_all("test/x")
    test_Y = load_all("test/y")
    test_groups = list(map(lambda v: v[()], hdf["test/groups"].values()))

# %%
X = train_X + test_X

scaler = sklearn.preprocessing.StandardScaler().fit(np.concatenate(X))
X = list(map(lambda x: scaler.transform(x), X))
X_train = list(map(lambda x: scaler.transform(x), train_X))

# Dimensions: (# frames) x (mouse ID) x (x, y coordinate) x (body part).
# Units: pixels; coordinates are relative to the entire image.
# Original image dimensions are 1024 x 570.

# %%
sample_lengths = np.array(list(map(len, X)))
p_draw = sample_lengths / np.sum(sample_lengths)

min(sample_lengths)

# %%
class CausalConv1D(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        downsample=False,
        **conv1d_kwargs,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.downsample = downsample
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.pad,
            dilation=dilation,
            **conv1d_kwargs,
        )

    def forward(self, x):
        # remove trailing padding
        x = self.conv(x)
        x = x[:, :, : -self.conv.padding[0]]

        if self.downsample:
            x = x[:, :, :: self.conv.dilation[0]]

        return x


class Embedder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0, **kwargs):
        super().__init__()

        self.reduction = 2 ** 5

        self.head = CausalConv1D(in_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.convolutions = torch.nn.ModuleList(
            [
                CausalConv1D(out_channels, out_channels, dilation=2),
                CausalConv1D(out_channels, out_channels, dilation=2),
                CausalConv1D(out_channels, out_channels, dilation=2),
                CausalConv1D(out_channels, out_channels, dilation=2),
                CausalConv1D(out_channels, out_channels, dilation=2),
                CausalConv1D(out_channels, out_channels),
            ]
        )

    def forward(self, x):
        x = self.head(x)
        for i, l in enumerate(self.convolutions):
            x_ = torch.nn.functional.leaky_relu(l(x))
            x = x + self.dropout(x_)

        return x


def nt_xent_loss(predictions_to, has_value_to=None, cpc_tau=0.07, cpc_alpha=0.5):
    # Normalized Temperature-scaled Cross Entropy Loss
    # https://arxiv.org/abs/2002.05709v3

    if has_value_to is not None:
        predictions_to = predictions_to[has_value_to]
        labels = torch.where(has_value_to)[0]
        labels_diag = torch.zeros(
            predictions_to.shape[0],
            predictions_to.shape[1],
            device=predictions_to.device,
            dtype=torch.bool,
        )
        labels_diag[torch.arange(predictions_to.shape[0]), labels] = True
    else:
        labels_diag = torch.diag(
            torch.ones(len(predictions_to), device=predictions_to.device)
        ).bool()

    neg = predictions_to[~labels_diag].reshape(len(predictions_to), -1)
    pos = predictions_to[labels_diag]

    neg_and_pos = torch.cat((neg, pos.unsqueeze(1)), dim=1)

    loss_pos = -pos / cpc_tau
    loss_neg = torch.logsumexp(neg_and_pos / cpc_tau, dim=1)

    loss = 2 * (cpc_alpha * loss_pos + (1.0 - cpc_alpha) * loss_neg)

    return loss


# %%
batch_size = 128
num_batches = 100000
subsample_length = 512
indices = np.arange(len(X))
num_features = X[0].shape[-1]
num_embeddings = 128
num_context = 128
num_ahead = 128
dropout = 0.1
device = "cuda:1"

# %%
embedder = Embedder(num_features, num_embeddings, dropout=dropout).to(device).train()
contexter = (
    torch.nn.GRU(num_embeddings, num_context, batch_first=True, dropout=dropout, num_layers=2)
    .to(device)
    .train()
)
projections = [
    torch.nn.Linear(num_context, num_embeddings, bias=False).to(device)
    for _ in range(num_ahead // 4)
]

params = list(itertools.chain.from_iterable(map(lambda p: list(p.parameters()), projections)))
params += list(embedder.parameters())
params += list(contexter.parameters())

optimizer = torch.optim.AdamW(params)

# %%
losses = []

# %%
embedder = embedder.train()
contexter = contexter.train()

bar = progress_bar(range(num_batches))
for i_batch in bar:
    optimizer.zero_grad()

    indices_batch = np.random.choice(indices, size=batch_size, p=p_draw)
    X_batch = [X[i].astype(np.float32) for i in indices_batch]

    padding = np.array([max(0, subsample_length + num_ahead - len(x)) for x in X_batch])

    X_batch = [np.pad(x, ((0, pad), (0, 0))) for x, pad in zip(X_batch, padding)]

    valid_lengths = np.array([len(x) - pad for x, pad in zip(X_batch, padding)])
    subsample_from_max = np.clip(valid_lengths - subsample_length - num_ahead, 0, np.inf)
    subsample_from = np.floor(np.random.rand(batch_size) * subsample_from_max).astype(np.int)

    X_batch = np.stack(
        [
            X_batch[i][subsample_from[i] : subsample_from[i] + subsample_length + num_ahead]
            for i in range(batch_size)
        ]
    )

    X_batch = torch.transpose(torch.from_numpy(X_batch), 2, 1).to(device)
    X_emb = embedder(X_batch)

    max_from_timesteps = [x.shape[-1] - num_ahead for x in X_emb]
    from_timesteps = np.random.randint(
        low=0,
        high=max_from_timesteps,
        size=batch_size,
    )

    max_context_timestep = max(from_timesteps)
    contexts, _ = contexter(X_emb[:, :, : max(from_timesteps) + 1].transpose(2, 1))

    # from_timesteps_reduced = np.floor(from_timesteps / embedder.reduction).astype(np.int)
    from_timesteps_reduced = from_timesteps
    contexts_from = torch.stack([contexts[i, from_timesteps_reduced[i]] for i in range(batch_size)])
    contexts_from = torch.nn.functional.normalize(contexts_from, p=2, dim=-1)

    ahead = 0
    batch_loss = []
    for ahead in range(1, num_ahead, 4):
        embeddings_projected = projections[ahead // 4](contexts_from)
        embeddings_to = torch.stack(
            [X_emb[i, :, from_timesteps_reduced[i] + ahead] for i in range(batch_size)]
        )
        embeddings_to = torch.nn.functional.normalize(embeddings_to, p=2, dim=-1)

        # assume both are l2-normalized -> cosine similarity
        predictions_to = torch.einsum("ae,ce->ac", contexts_from, embeddings_to)

        has_value_to = torch.from_numpy((from_timesteps_reduced + ahead) < valid_lengths).to(device)
        loss = nt_xent_loss(predictions_to, has_value_to)
        loss_mean = torch.mean(loss)

        batch_loss.append(loss_mean)

    batch_loss = sum(batch_loss) / len(batch_loss)

    batch_loss.backward()
    optimizer.step()

    losses.append(batch_loss.cpu().item())
    bar.comment = f"{np.mean(losses[-100:])}"

# %%
plt.plot(np.arange(len(losses) - 100) + 100, losses[100:], alpha=0.1)
plt.plot(pd.Series(losses).rolling(100).mean())

# %%
torch.save((embedder, contexter, projections, losses), mabe.config.ROOT_PATH / "cpc_model_2.pt")

# %%
embedder, contexter, projections, losses = torch.load(mabe.config.ROOT_PATH / "cpc_model_2.pt")

# %%
embedder = embedder.eval()
contexter = contexter.eval()

contexts = []
with torch.no_grad():
    bar = progress_bar(range(len(X_train)))
    for idx in bar:
        x = X_train[idx].astype(np.float32)
        x = torch.transpose(torch.from_numpy(x[None, :, :]), 2, 1).to(device)
        x_emb = embedder(x)

        c, _ = contexter(x_emb.transpose(2, 1))
        c = torch.nn.functional.normalize(c, p=2, dim=-1)

        contexts.append(c.cpu().numpy()[0])

train_contexts = np.concatenate(contexts)

# %%
train_Y_flat = np.concatenate(train_Y)
train_X_flat = np.concatenate(X_train)
train_groups = np.concatenate([np.ones(len(X_train[i])) * i for i in range(len(X_train))]).astype(
    np.int
)

# %%
linear = sklearn.linear_model.LogisticRegression(multi_class="multinomial", max_iter=1000, C=1e20)
linear.fit(train_contexts, train_Y_flat)
train_Y_pred = linear.predict(train_contexts)

sklearn.metrics.f1_score(
    train_Y_flat, train_Y_pred, average="macro"
), sklearn.metrics.precision_score(train_Y_flat, train_Y_pred, average="macro")

# %%
linear = sklearn.linear_model.LogisticRegression(multi_class="multinomial", max_iter=1000)
scores = sklearn.model_selection.cross_validate(
    linear,
    train_contexts,
    train_Y_flat,
    n_jobs=-1,
    cv=sklearn.model_selection.GroupShuffleSplit(32),
    groups=train_groups,
    scoring=dict(
        f1=sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average="macro"),
        precision=sklearn.metrics.make_scorer(sklearn.metrics.precision_score, average="macro"),
    ),
)

np.median(scores["test_f1"]), np.median(scores["test_precision"])

# %%
linear = sklearn.linear_model.LogisticRegression(multi_class="multinomial", max_iter=1000)
scores = sklearn.model_selection.cross_validate(
    linear,
    train_X_flat,
    train_Y_flat,
    n_jobs=-1,
    cv=sklearn.model_selection.GroupShuffleSplit(32),
    groups=train_groups,
    scoring=dict(
        f1=sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average="macro"),
        precision=sklearn.metrics.make_scorer(sklearn.metrics.precision_score, average="macro"),
    ),
)

np.median(scores["test_f1"]), np.median(scores["test_precision"])
