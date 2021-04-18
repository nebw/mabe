from typing import Optional

import numpy as np
import sklearn
import sklearn.decomposition
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import torch
from fastprogress.fastprogress import force_console_behavior

import mabe
import mabe.config
import mabe.features
import mabe.model

master_bar, progress_bar = force_console_behavior()

# %%
device = "cuda:3"

# %%
result_file = "training_results_2021-04-16 05:26:50.104528_baseline2_task12_smallcontext_0.845.pt"
# TODO: use all runs
result = torch.load(mabe.config.ROOT_PATH / result_file)[0]

# %%
config = result.config
cpc_params = result.best_params[0]
num_features = 37
num_extra_features = 2

cpc = mabe.model.ConvCPC(
    num_features,
    config.num_embeddings,
    config.num_context,
    config.num_ahead,
    config.num_ahead_subsampling,
    config.subsample_length,
    num_embedder_blocks=config.num_embedder_blocks,
    input_dropout=config.input_dropout,
    head_dropout=config.head_dropout,
    dropout=config.dropout,
    split_idx=config.split_idx,
    num_extra_features=num_extra_features,
).to(device)

cpc.load_state_dict(cpc_params)
cpc = cpc.eval()

# %%
task3_path = mabe.config.ROOT_PATH / "train_task3.npy"
test_path = mabe.config.ROOT_PATH / "test-release.npy"

# %%
X_test, X_test_extra, _, groups_test, _ = mabe.features.load_dataset(test_path)

features_test = []
with torch.no_grad():
    for idx in range(len(X_test)):
        # from feature preprocessing
        crop_pre = 1
        crop_post = 0

        group = groups_test[idx]
        x = X_test[idx].astype(np.float32)
        if config.use_extra_features:
            x_extra = X_test_extra[idx].astype(np.float32)
            x_extra = torch.from_numpy(x_extra).to(device, non_blocking=True)

        x = torch.transpose(torch.from_numpy(x[None, :, :]), 2, 1).to(device, non_blocking=True)
        x_emb = cpc.embedder(x)

        crop = (x.shape[-1] - x_emb.shape[-1]) // 2
        crop_pre += crop
        crop_post += crop

        c = cpc.apply_contexter(x_emb, device)

        crop = x_emb.shape[-1] - c.shape[-1]
        crop_pre += crop

        logreg_features = c[0].T
        if config.use_extra_features:
            x_extra = x_extra[crop_pre : -(crop_post - 1)]

        if config.use_extra_features:
            x_cominbed = torch.cat((logreg_features, x_extra), dim=-1)
        else:
            x_combined = logreg_features

        x_combined = x_cominbed.cpu().data.numpy()
        features_test.append(x_combined)
features_test = np.concatenate(features_test)

# %%
cv_scores = []
for behavior, (X, X_extra, Y, groups, annotators) in mabe.features.load_task3_datasets(task3_path):
    X_flat = []
    Y_flat = []
    groups_flat = []
    with torch.no_grad():
        for idx in range(len(X)):
            # from feature preprocessing
            crop_pre = 1
            crop_post = 0

            x = X[idx].astype(np.float32)
            x_extra = None
            if config.use_extra_features:
                x_extra = X_extra[idx].astype(np.float32)
                x_extra = torch.from_numpy(x_extra).to(device, non_blocking=True)

            g = np.array([idx])

            x = torch.transpose(torch.from_numpy(x[None, :, :]), 2, 1).to(device, non_blocking=True)
            x_emb = cpc.embedder(x)

            crop = (x.shape[-1] - x_emb.shape[-1]) // 2
            crop_pre += crop
            crop_post += crop

            c = cpc.apply_contexter(x_emb, device)

            crop = x_emb.shape[-1] - c.shape[-1]
            crop_pre += crop

            logreg_features = c[0].T
            x_extra = x_extra[crop_pre : -(crop_post - 1)]
            y = Y[idx][crop_pre : -(crop_post - 1)]

            x_cominbed = torch.cat((logreg_features, x_extra), dim=-1)

            X_flat.append(x_cominbed.cpu().data.numpy())
            Y_flat.append(y)
            groups_flat.append(g.repeat(len(y)))

        X_flat = np.concatenate(X_flat)
        Y_flat = np.concatenate(Y_flat)
        groups_flat = np.concatenate(groups_flat)

    print(behavior)
    print(len(np.unique(groups_flat)))
    if len(np.unique(groups_flat)) > 1:
        cv = sklearn.model_selection.GroupShuffleSplit(8)
    else:
        cv = sklearn.model_selection.StratifiedShuffleSplit(8)

    X_flat_all = np.concatenate((X_flat, features_test))
    scaler = sklearn.preprocessing.StandardScaler().fit(X_flat_all)
    X_flat = scaler.transform(X_flat)

    linear = sklearn.pipeline.make_pipeline(
        sklearn.linear_model.LogisticRegression(
            multi_class="multinomial", class_weight="balanced", max_iter=1000, C=1e-1
        )
    )
    scores = sklearn.model_selection.cross_validate(
        linear,
        X_flat,
        Y_flat,
        n_jobs=8,
        cv=cv,
        groups=groups_flat,
        scoring=dict(
            f1=sklearn.metrics.make_scorer(sklearn.metrics.f1_score),  # , average="macro"),
            precision=sklearn.metrics.make_scorer(
                sklearn.metrics.precision_score
            ),  # , average="macro"),
        ),
    )
    if len(np.unique(groups_flat)) > 1:
        cv_scores.append(scores["test_f1"])
    print(np.median(scores["test_f1"]))
    print()

print(np.mean(cv_scores))


# %%
submission: dict[str, dict] = {}
for behavior, (X, X_extra, Y, groups, annotators) in mabe.features.load_task3_datasets(task3_path):
    submission[behavior] = dict()
    X_flat = []
    Y_flat = []
    groups_flat = []
    with torch.no_grad():
        for idx in range(len(X)):
            # from feature preprocessing
            crop_pre = 1
            crop_post = 0

            x = X[idx].astype(np.float32)
            x_extra = None
            if config.use_extra_features:
                x_extra = X_extra[idx].astype(np.float32)
                x_extra = torch.from_numpy(x_extra).to(device, non_blocking=True)

            g = np.array([idx])

            x = torch.transpose(torch.from_numpy(x[None, :, :]), 2, 1).to(device, non_blocking=True)
            x_emb = cpc.embedder(x)

            crop = (x.shape[-1] - x_emb.shape[-1]) // 2
            crop_pre += crop
            crop_post += crop

            c = cpc.apply_contexter(x_emb, device)

            crop = x_emb.shape[-1] - c.shape[-1]
            crop_pre += crop

            logreg_features = c[0].T
            x_extra = x_extra[crop_pre : -(crop_post - 1)]
            y = Y[idx][crop_pre : -(crop_post - 1)]

            x_cominbed = torch.cat((logreg_features, x_extra), dim=-1)

            X_flat.append(x_cominbed.cpu().data.numpy())
            Y_flat.append(y)
            groups_flat.append(g.repeat(len(y)))

        X_flat = np.concatenate(X_flat)
        Y_flat = np.concatenate(Y_flat)
        groups_flat = np.concatenate(groups_flat)

    X_flat_all = np.concatenate((X_flat, features_test))
    scaler = sklearn.preprocessing.StandardScaler().fit(X_flat_all)
    X_flat = scaler.transform(X_flat)

    linear = sklearn.pipeline.make_pipeline(
        sklearn.linear_model.LogisticRegression(
            multi_class="multinomial", class_weight="balanced", max_iter=1000, C=1e-1
        )
    )
    linear.fit(X_flat, Y_flat)

    with torch.no_grad():
        for idx in range(len(X_test)):
            # from feature preprocessing
            crop_pre = 1
            crop_post = 0

            group = groups_test[idx]
            x = X_test[idx].astype(np.float32)
            x_extra = None
            if config.use_extra_features:
                x_extra = X_test_extra[idx].astype(np.float32)
                x_extra = torch.from_numpy(x_extra).to(device, non_blocking=True)

            x = torch.transpose(torch.from_numpy(x[None, :, :]), 2, 1).to(device, non_blocking=True)
            x_emb = cpc.embedder(x)

            crop = (x.shape[-1] - x_emb.shape[-1]) // 2
            crop_pre += crop
            crop_post += crop

            c = cpc.apply_contexter(x_emb, device)

            crop = x_emb.shape[-1] - c.shape[-1]
            crop_pre += crop

            logreg_features = c[0].T
            x_extra = x_extra[crop_pre : -(crop_post - 1)]

            x_cominbed = torch.cat((logreg_features, x_extra), dim=-1)
            x_combined = x_cominbed.cpu().data.numpy()

            y_pred = linear.predict(scaler.transform(x_combined))
            # TODO: off-by-one?
            y_pred = np.concatenate(
                (y_pred[:1].repeat(crop_pre), y_pred, y_pred[-1:].repeat(crop_post))
            )

            submission[behavior][group] = y_pred

# %%
sample_submission = np.load(
    mabe.config.ROOT_PATH / "sample-submission-task3.npy", allow_pickle=True
).item()


def validate_submission(submission, sample_submission):
    if not isinstance(submission, dict):
        print("Submission should be dict")
        return False

    if not submission.keys() == sample_submission.keys():
        print("Submission keys don't match")
        return False
    for behavior in submission:
        sb = submission[behavior]
        ssb = sample_submission[behavior]
        if not isinstance(sb, dict):
            print("Submission should be dict")
            return False

        if not sb.keys() == ssb.keys():
            print("Submission keys don't match")
            return False

        for key in sb:
            sv = sb[key]
            ssv = ssb[key]
            if not len(sv) == len(ssv):
                print(f"Submission lengths of {key} doesn't match")
                return False

        for key, sv in sb.items():
            if not all(isinstance(x, (np.int32, np.int64, int)) for x in list(sv)):
                print(f"Submission of {key} is not all integers")
                return False

    print("All tests passed")
    return True


# %%
if validate_submission(submission, sample_submission):
    np.save(mabe.config.ROOT_PATH / "task3_submission2.npy", submission)
