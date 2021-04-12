#!/usr/bin/env python

import copy
import datetime

import click
import click_pathlib
import madgrad
import numpy as np
import sklearn.utils
import torch
from fastprogress.fastprogress import progress_bar

import mabe
import mabe.config
import mabe.data
import mabe.loss
import mabe.model
import mabe.ringbuffer
import mabe.training


def predict_test_data(cpc, logreg, data, device):
    cpc = cpc.eval()
    logreg = logreg.eval()

    test_predictions = {}
    test_logits = {}
    with torch.no_grad():
        bar = progress_bar(range(len(data.X_test)))
        for idx in bar:
            # from feature preprocessing
            crop_pre = 1
            crop_post = 0

            x = data.X_test[idx].astype(np.float32)
            g = data.test_groups[idx]

            x = torch.transpose(torch.from_numpy(x[None, :, :]), 2, 1).to(device, non_blocking=True)
            x_emb = cpc.embedder(x)

            crop = (x.shape[-1] - x_emb.shape[-1]) // 2
            crop_pre += crop
            crop_post += crop

            c = cpc.apply_contexter(x_emb, device)
            l = logreg(c[0].T)

            crop = x_emb.shape[-1] - c.shape[-1]
            crop_pre += crop

            l = torch.cat((l[:1].repeat(crop_pre, 1), l, l[-1:].repeat(crop_post, 1)), dim=0)
            p = torch.argmax(l, dim=-1)

            assert len(p) == x.shape[-1] + 1

            test_predictions[g.decode("utf-8")] = p.cpu().data.numpy()
            test_logits[g.decode("utf-8")] = l.cpu().data.numpy()

    return test_predictions, test_logits


def train_model(config, data, split, device):
    batches_per_epoch = int(sum(data.sample_lengths) / config.subsample_length / config.batch_size)
    num_features = data.X[0].shape[-1]

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
    ).to(device)

    logreg = torch.nn.Sequential(
        torch.nn.LayerNorm(config.num_context),
        torch.nn.Linear(config.num_context, 4),
    )
    logreg = logreg.to(device)

    optimizer = None
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            list(cpc.parameters()) + list(logreg.parameters()),
            weight_decay=config.weight_decay,
            lr=config.learning_rate,
            momentum=0.9,
            nesterov=True,
        )
    elif config.optimizer == "MADGRAD":
        optimizer = madgrad.MADGRAD(
            list(cpc.parameters()) + list(logreg.parameters()),
            weight_decay=config.weight_decay,
            lr=config.learning_rate,
        )
    assert optimizer is not None

    class_weights = sklearn.utils.class_weight.compute_class_weight(
        "balanced",
        classes=np.unique(np.concatenate(data.train_Y).astype(np.int)),
        y=np.concatenate(data.train_Y).astype(np.int),
    )

    clf_loss = mabe.loss.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(device).float(),
        ignore_index=-1,
        smooth_eps=config.label_smoothing,
        smooth_dist=torch.from_numpy(split.p_class).to(device).float(),
    ).to(device)

    scheduler = None
    if config.scheduler == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.num_epochs * batches_per_epoch
        )
    elif config.scheduler == "none":
        pass
    else:
        assert False

    losses = []
    clf_losses = []
    clf_val_f1s = []
    best_params = None

    best_val_f1 = mabe.training.validation_f1(cpc, logreg, data, split, device)
    clf_val_f1s.append(best_val_f1)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    running = lambda l: np.mean(l[-batches_per_epoch:])
    # use combined bar for epochs and batches
    bar = progress_bar(range(config.num_epochs * batches_per_epoch))
    bar_iter = iter(bar)
    for i_epoch in range(config.num_epochs):
        for _ in range(batches_per_epoch):
            next(bar_iter)

            optimizer.zero_grad()
            cpc = cpc.train()

            X_batch, Y_batch, indices_batch = split.get_train_batch(
                config.batch_size, random_noise=config.augmentation_random_noise
            )

            contexts, Y_batch, batch_loss = cpc(X_batch, Y_batch, device=device, with_loss=True)
            losses.append(batch_loss.cpu().item())

            has_train_labels = [i in split.train_indices_labeled for i in indices_batch]
            Y_batch_flat = Y_batch[has_train_labels].flatten().long()
            valids = Y_batch_flat >= 0
            if torch.any(valids):
                clf_batch_loss = clf_loss(
                    logreg(contexts[has_train_labels]).reshape(-1, 4),
                    Y_batch[has_train_labels].flatten().long(),
                )

                clf_losses.append(clf_batch_loss.cpu().item())
                batch_loss = batch_loss + config.clf_loss_scaling * clf_batch_loss

                batch_loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                bar.comment = f"Train: {running(losses):.3f} | CLF Train: {running(clf_losses):.3f} | Best CLF Val F1: {best_val_f1:.3f} | LR: {get_lr(optimizer):.4f}"

        val_f1 = mabe.training.validation_f1(cpc, logreg, data, split, device)
        clf_val_f1s.append(val_f1)

        get_cpu_params = lambda model: copy.deepcopy(
            {k: v.cpu() for k, v in model.state_dict().items()}
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_params = (get_cpu_params(cpc), get_cpu_params(logreg))

    test_predictions, test_logits = predict_test_data(cpc, logreg, data, device)

    result = mabe.training.TrainingResult(
        config=config,
        losses=losses,
        clf_losses=clf_losses,
        clf_val_f1s=clf_val_f1s,
        best_val_f1=best_val_f1,
        best_params=best_params,
        test_predictions=test_predictions,
        test_logits=test_logits,
    )

    return result


def parse_optional_parameter(val: str):
    try:
        val = int(val)
    except ValueError:
        try:
            val = float(val)
        except ValueError:
            pass
    return val


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("--model_name", type=str, required=True)
@click.option("--device", type=str, required=True)
@click.option(
    "--feature_path",
    type=click_pathlib.Path(exists=True),
    default=mabe.config.ROOT_PATH / "features.hdf5",
)
@click.pass_context
def train(click_context, model_name, device, feature_path):
    config_kwargs = {}
    for arg in click_context.args:
        key, val = arg.split("--")[1].split("=")
        config_kwargs[key] = parse_optional_parameter(val)

    data = mabe.data.DataWrapper(feature_path)
    results = []
    for split_idx in range(10):
        config = mabe.training.TrainingConfig(
            split_idx=split_idx, feature_path=feature_path, **config_kwargs
        )
        split = mabe.data.CVSplit(split_idx, data)
        result = train_model(config, data, split, device)
        print(f"\nBest validation F1 (split {split_idx}): {result.best_val_f1:.3f}")

        results.append(result)

    f1s = [r.best_val_f1 for r in results]
    print(f"Validation F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    torch.save(
        results,
        mabe.config.ROOT_PATH
        / f"training_results_{datetime.datetime.now()}_{model_name}_{np.mean(f1s):.3f}.pt",
    )


if __name__ == "__main__":
    train()
