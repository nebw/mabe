import dataclasses
import pathlib

import numpy as np
import torch

import mabe.loss


def validation_f1(cpc, logreg, data, split, device, config):
    with torch.no_grad():
        cpc = cpc.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for idx in split.val_indices_labeled:
                x = data.X_train[idx].astype(np.float32)
                if config.use_extra_features:
                    x_extra = data.X_extra_train[idx].astype(np.float32)
                    x_extra = torch.from_numpy(x_extra).to(device, non_blocking=True)
                y = data.train_Y[idx]

                x = torch.transpose(torch.from_numpy(x[None, :, :]), 2, 1).to(
                    device, non_blocking=True
                )
                x_emb = cpc.embedder(x)

                crop = (y.shape[-1] - x_emb.shape[-1]) // 2
                y = y[crop:-crop]
                x_extra = x_extra[crop:-crop]

                c = cpc.apply_contexter(x_emb, device)

                crop = len(y) - c.shape[-1]
                y = y[crop:]
                x_extra = x_extra[crop:]

                logreg_features = c[0].T
                if config.use_extra_features:
                    logreg_features = torch.cat((logreg_features, x_extra), dim=-1)

                l = logreg(logreg_features)
                p = torch.argmax(l, dim=-1)

                predictions.append(p.cpu().numpy())
                labels.append(y)

    predictions = np.concatenate(predictions).astype(np.int)
    labels = np.concatenate(labels).astype(np.int)

    return mabe.loss.macro_f1_score(labels, predictions, 4)


@dataclasses.dataclass
class TrainingConfig:
    split_idx: int
    feature_path: pathlib.Path = mabe.config.ROOT_PATH / "features.hdf5"
    batch_size: int = 32
    num_epochs: int = 40
    subsample_length: int = 256
    num_embeddings: int = 128
    num_context: int = 512
    num_ahead: int = 32 * 8
    num_ahead_subsampling: int = 32
    num_embedder_blocks: int = 3
    input_dropout: float = 0.0
    head_dropout: float = 0.0
    dropout: float = 0.0
    clf_loss_scaling: float = 1.0
    label_smoothing: float = 0.2
    optimizer: str = "SGD"
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    scheduler: str = "cosine_annealing"
    augmentation_random_noise: float = 0.0
    use_extra_features: bool = False


@dataclasses.dataclass
class TrainingResult:
    config: TrainingConfig
    losses: list
    clf_losses: list
    clf_val_f1s: list
    best_val_f1: float
    best_params: tuple
    test_predictions: np.array
    test_logits: np.array
