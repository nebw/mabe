#!/usr/bin/env python

import datetime
from typing import Union

import click
import click_pathlib
import numpy as np
import torch

import mabe
import mabe.config
import mabe.data
import mabe.loss
import mabe.model
import mabe.ringbuffer
import mabe.training
import mabe.util


def parse_optional_parameter(val: Union[str, int, float]) -> Union[str, int, float]:
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
@click.option("--device", type=str, required=True)
@click.option(
    "--feature_path",
    type=click_pathlib.Path(exists=True),
    default=mabe.config.ROOT_PATH / "features.hdf5",
)
@click.option("--from_split", type=int, default=0)
@click.option("--to_split", type=int, default=10)
@click.pass_context
def train(click_context, model_name, device, feature_path, from_split, to_split):
    config_kwargs = {}
    for arg in click_context.args:
        key, val = arg.split("--")[1].split("=")
        config_kwargs[key] = parse_optional_parameter(val)

    data = mabe.data.DataWrapper(feature_path)
    results = []
    for split_idx in range(from_split, to_split):
        config = mabe.training.TrainingConfig(
            split_idx=split_idx, feature_path=feature_path, **config_kwargs
        )
        split = mabe.data.CVSplit(split_idx, data)
        trainer = mabe.training.Trainer(config, data, split, device)
        result = trainer.train_model()
        print(f"\nBest validation F1 (split {split_idx}): {result.best_val_f1[0]:.3f}")

        results.append(result)

    f1s = [r.best_val_f1[0] for r in results]
    print(f"Validation F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    torch.save(
        results,
        mabe.config.ROOT_PATH
        / f"training_results_{datetime.datetime.now()}_{model_name}_{np.mean(f1s):.3f}.pt",
        _use_new_zipfile_serialization=False,
    )


if __name__ == "__main__":
    train()
