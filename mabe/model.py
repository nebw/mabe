import math
from typing import Tuple

import numba
import numpy as np
import torch
import torchtyping
from fastprogress.fastprogress import force_console_behavior
from torchtyping import TensorType  # type: ignore

import mabe
import mabe.data
from mabe.types import batch, channels, time

torchtyping.patch_typeguard()
master_bar, progress_bar = force_console_behavior()


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

    def forward(
        self, x: TensorType["batch", "channels", "time", float]
    ) -> TensorType["batch", "channels", "time", float]:
        x = self.conv(x)

        # remove trailing padding
        x = x[:, :, self.pad : -self.pad]

        if self.downsample:
            x = x[:, :, :: self.conv.dilation[0]]

        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, padding, kernel_size, layer_norm=False, causal=False, **kwargs):
        super().__init__()

        # TODO: add dilation
        self.offset = (kernel_size - 1) // 2
        self.causal = causal

        self.conv1 = torch.nn.Conv1d(channels, channels, padding=padding, kernel_size=kernel_size)
        self.conv2 = torch.nn.Conv1d(channels, channels, padding=padding, kernel_size=kernel_size)

        self.layer_norm = layer_norm
        self.layer_norm1 = torch.nn.LayerNorm(channels) if layer_norm else None
        self.layer_norm2 = torch.nn.LayerNorm(channels) if layer_norm else None

    def forward(
        self, x: TensorType["batch", "channels", "time", float]
    ) -> TensorType["batch", "channels", "time", float]:
        x_ = x

        if self.layer_norm:
            x = x.transpose(2, 1)
            x = self.layer_norm1(x)
            x = x.transpose(2, 1)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv1(x)

        if self.layer_norm:
            x = x.transpose(2, 1)
            x = self.layer_norm2(x)
            x = x.transpose(2, 1)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv2(x)

        if self.offset > 0:
            if self.causal:
                x_ = x_[:, :, 4 * self.offset :]
            else:
                x_ = x_[:, :, 2 * self.offset : -2 * self.offset]
        return x_ + x


class Embedder(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        layer_norm=True,
        input_dropout=0,
        head_dropout=0,
        num_embedder_blocks=3,
        **kwargs,
    ):
        super().__init__()

        self.input_dropout = torch.nn.Dropout2d(p=input_dropout)
        self.head_dropout = torch.nn.Dropout2d(p=head_dropout)

        self.dropout = torch.nn.Dropout(input_dropout)
        self.head = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

        self.convolutions = torch.nn.Sequential(
            ResidualBlock(out_channels, kernel_size=1, padding=0, layer_norm=layer_norm),
            ResidualBlock(out_channels, kernel_size=1, padding=0, layer_norm=layer_norm),
            *[
                ResidualBlock(out_channels, kernel_size=3, padding=0, layer_norm=layer_norm)
                for _ in range(num_embedder_blocks)
            ],
        )

    def forward(
        self, x: TensorType["batch", "channels", "time", float]
    ) -> TensorType["batch", "channels", "time", float]:
        x = self.input_dropout(x)
        x = self.head(x)
        x = self.head_dropout(x)

        x = self.convolutions(x)

        return x


class CausalResidualBlock(torch.nn.Module):
    def __init__(self, channels, layer_norm=False, **kwargs):
        super().__init__()

        self.conv1 = CausalConv1D(channels, channels, dilation=2)
        self.conv2 = CausalConv1D(channels, channels, dilation=1)

        self.layer_norm = layer_norm
        self.layer_norm1 = torch.nn.LayerNorm(channels) if layer_norm else None
        self.layer_norm2 = torch.nn.LayerNorm(channels) if layer_norm else None

    def forward(
        self, x: TensorType["batch", "channels", "time", float]
    ) -> TensorType["batch", "channels", "time", float]:
        x_ = x

        if self.layer_norm:
            x = x.transpose(2, 1)
            x = self.layer_norm1(x)
            x = x.transpose(2, 1)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv1(x)

        if self.layer_norm:
            x = x.transpose(2, 1)
            x = self.layer_norm2(x)
            x = x.transpose(2, 1)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv2(x)

        return x_ + x


class Contexter(torch.nn.Module):
    def __init__(self, in_channels, out_channels, layer_norm=True, dropout=0, **kwargs):
        super().__init__()

        self.head = CausalConv1D(in_channels, out_channels)
        self.convolutions = torch.nn.Sequential(
            *[
                ResidualBlock(
                    out_channels, kernel_size=3, padding=0, layer_norm=layer_norm, causal=True
                )
                for i in range(8)
            ]
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, x: TensorType["batch", "channels", "time", float]
    ) -> TensorType["batch", "channels", "time", float]:
        x = self.head(x)
        x = self.convolutions(x)
        x = self.dropout(x)

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


# @numba.njit
def subsample(X, X_, valid_lengths, combined_length, subsample_from_rand):
    for i, x in enumerate(X):
        from_idx = math.floor(subsample_from_rand[i] * max(0, len(x) - combined_length))
        to_idx = min(len(x), from_idx + combined_length)
        valid_length = to_idx - from_idx
        sample = x[from_idx:to_idx]
        X_[i, :valid_length] = sample
        valid_lengths[i] = valid_length


class ConvCPC(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_embeddings,
        num_context,
        num_ahead,
        num_ahead_subsampling,
        subsample_length,
        split_idx,
        num_embedder_blocks=3,
        input_dropout=0,
        head_dropout=0,
        dropout=0,
        num_extra_features=0,
        **kwargs,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.num_context = num_context
        self.num_features = num_features
        self.num_extra_features = num_extra_features
        self.split_idx = split_idx

        self.embedder = Embedder(
            num_features,
            num_embeddings,
            num_embedder_blocks=num_embedder_blocks,
            input_dropout=input_dropout,
            head_dropout=head_dropout,
            dropout=dropout,
        )
        self.contexter = Contexter(num_embeddings, num_context, dropout=dropout)

        self.projections = torch.nn.ModuleList(
            [
                torch.nn.Linear(num_context, num_embeddings, bias=False)
                for _ in range(num_ahead // num_ahead_subsampling)
            ]
        )

        self.num_ahead = num_ahead
        self.num_ahead_subsampling = num_ahead_subsampling
        self.subsample_length = subsample_length

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.crop_pre = None
        self.crop_post = None

    def get_crops(self, device):
        if self.crop_pre is None:
            with torch.no_grad():
                initial_length = 128
                x_dummy = torch.zeros(1, self.num_features, initial_length, device=device)
                x_postemb = self.embedder(x_dummy)

                crop = (initial_length - x_postemb.shape[-1]) // 2
                self.crop_pre = crop
                self.crop_post = crop

                x_postcontext = self.apply_contexter(x_postemb, device)

                crop = x_postemb.shape[-1] - x_postcontext.shape[-1]
                self.crop_pre += crop

        return self.crop_pre, self.crop_post

    def random_subsample(
        self,
        batch: mabe.data.TrainingBatch,
        subsample_from_rand: TensorType["batch"] = None,
    ) -> Tuple[
        TensorType["batch", "time", "channels", float],
        TensorType["batch", "time", "channels", float],
        TensorType["batch", "time", int],
        TensorType["batch", "time", int],
        TensorType["batch", int],
    ]:
        batch_size = len(batch.X)
        if subsample_from_rand is None:
            subsample_from_rand = np.random.rand(batch_size)
        combined_length = self.subsample_length + self.num_ahead

        X_ = np.zeros((batch_size, combined_length, self.num_features), dtype=np.float32)
        valid_lengths = np.zeros(batch_size, dtype=np.int)
        subsample(batch.X, X_, valid_lengths, combined_length, subsample_from_rand)

        X_extra_ = None
        if batch.X_extra is not None:
            X_extra_ = np.zeros(
                (batch_size, combined_length, self.num_extra_features), dtype=np.float32
            )
            subsample(
                batch.X_extra,
                X_extra_,
                valid_lengths,
                combined_length,
                subsample_from_rand,
            )

        Y_ = None
        if batch.Y is not None:
            Y_ = np.full((batch_size, combined_length), -1, dtype=np.int)
            subsample(batch.Y, Y_, valid_lengths, combined_length, subsample_from_rand)

        annotators_ = None
        annotators_ = np.full((batch_size, combined_length), -1, dtype=np.int)
        subsample(
            batch.annotators,
            annotators_,
            valid_lengths,
            combined_length,
            subsample_from_rand,
        )

        return X_, X_extra_, Y_, annotators_, valid_lengths

    def subsample_and_pad(
        self,
        batch: mabe.data.TrainingBatch,
        device: str = "cpu",
        subsample_from_rand: TensorType["batch", int] = None,
    ) -> Tuple[
        TensorType["batch", "time", "channels", float],
        TensorType["batch", "time", "channels", float],
        TensorType["batch", "time", int],
        TensorType["batch", "time", int],
        TensorType["batch", int],
    ]:
        X, X_extra, Y, annotators, valid_lengths = self.random_subsample(
            batch, subsample_from_rand=subsample_from_rand
        )
        X = torch.transpose(torch.from_numpy(X), 2, 1).to(device, non_blocking=True)

        if X_extra is not None:
            X_extra = torch.from_numpy(X_extra).to(device, non_blocking=True)

        if Y is not None:
            Y = torch.from_numpy(Y).to(device, non_blocking=True)

        return X, X_extra, Y, annotators, valid_lengths

    def cpc_loss(
        self,
        X_emb: TensorType["batch", "time", "channels", float],
        contexts: TensorType["batch", "time", "channels", float],
        from_timesteps: TensorType["batch", int],
        valid_lengths: TensorType["batch", int],
    ) -> TensorType[float]:
        batch_size = len(contexts)

        from_timesteps_reduced = from_timesteps
        contexts_from = torch.stack(
            [contexts[i, from_timesteps_reduced[i]] for i in range(batch_size)]
        )

        embeddings_projections = torch.stack(
            list(map(lambda p: p(contexts_from), self.projections))
        )
        embeddings_projections = torch.nn.functional.normalize(embeddings_projections, p=2, dim=-1)

        ahead = torch.arange(self.num_ahead_subsampling, self.num_ahead, self.num_ahead_subsampling)
        to_timesteps = torch.from_numpy(from_timesteps_reduced)[:, None] + ahead

        X_emb_t = X_emb.transpose(2, 1)
        embeddings_to = X_emb_t[
            torch.arange(batch_size)[:, None].repeat(1, len(ahead)).flatten(),
            to_timesteps.flatten(),
        ].reshape(batch_size, len(ahead), -1)
        embeddings_to = torch.nn.functional.normalize(embeddings_to, p=2, dim=-1)

        embeddings_projections = torch.stack(
            list(map(lambda p: p(contexts_from), self.projections[1:]))
        )
        embeddings_projections = torch.nn.functional.normalize(embeddings_projections, p=2, dim=-1)

        # assume both are l2-normalized -> cosine similarity
        predictions_to = torch.einsum("tac,btc->tab", embeddings_projections, embeddings_to)

        batch_loss = []
        for idx, ahead in enumerate(
            range(self.num_ahead_subsampling, self.num_ahead, self.num_ahead_subsampling)
        ):
            predictions_to_ahead = predictions_to[idx]

            has_value_to = torch.from_numpy((from_timesteps_reduced + ahead) < valid_lengths).to(
                X_emb.device, non_blocking=True
            )
            loss = nt_xent_loss(predictions_to_ahead, has_value_to)
            loss_mean = torch.mean(loss)

            batch_loss.append(loss_mean)

        aggregated_batch_loss = sum(batch_loss) / len(batch_loss)

        return aggregated_batch_loss

    def apply_contexter(self, X_emb, device="cpu"):
        contexts = self.contexter(X_emb)

        return contexts

    def get_contexts(self, X, device="cpu"):
        contexts = []
        with torch.no_grad():
            bar = progress_bar(range(len(X)))
            for idx in bar:
                x = X[idx].astype(np.float32)

                x = torch.transpose(torch.from_numpy(x[None, :, :]), 2, 1).to(
                    device, non_blocking=True
                )
                x_emb = self.embedder(x)
                c = self.apply_contexter(x_emb, device)

                contexts.append(c.cpu().numpy())

        contexts = np.concatenate(contexts)
        return contexts

    def add_padding_to_batch(self, batch: mabe.data.TrainingBatch, device: str):
        crop_pre, crop_post = self.get_crops(device)

        def concat_padding_1d(sequences):
            l = numba.typed.List()
            for x in sequences:
                l.append(
                    np.concatenate(
                        (
                            np.zeros((crop_pre,), dtype=x.dtype),
                            x,
                            np.zeros((crop_post,), dtype=x.dtype),
                        ),
                        axis=-1,
                    )
                )
            return l

        def concat_padding_2d(sequences):
            l = numba.typed.List()
            for x in sequences:
                l.append(
                    np.concatenate(
                        (
                            np.zeros((crop_pre, x.shape[1]), dtype=x.dtype),
                            x,
                            np.zeros((crop_post, x.shape[1]), dtype=x.dtype),
                        ),
                        axis=0,
                    )
                )
            return l

        batch.X = concat_padding_2d(batch.X)
        batch.X_extra = concat_padding_2d(batch.X_extra)
        batch.Y = concat_padding_1d(batch.Y)
        batch.annotators = concat_padding_1d(batch.annotators)

        return batch

    def forward(
        self,
        batch: mabe.data.TrainingBatch,
        device: str = "cpu",
        with_loss: bool = False,
        min_from_timesteps: int = 0,
        subsample_from_rand: TensorType["batch", int] = None,
    ):
        batch_size = len(batch.X)
        batch = self.add_padding_to_batch(batch, device)

        (
            X_batch_samples,
            X_extra_batch_samples,
            Y_batch_samples,
            annotators_samples,
            valid_lengths,
        ) = self.subsample_and_pad(batch, device, subsample_from_rand=subsample_from_rand)

        X_emb = self.embedder(X_batch_samples)
        crop = (Y_batch_samples.shape[-1] - X_emb.shape[-1]) // 2
        valid_lengths -= crop
        if X_extra_batch_samples is not None:
            X_extra_batch_samples = X_extra_batch_samples[:, crop:-crop]
        if Y_batch_samples is not None:
            Y_batch_samples = Y_batch_samples[:, crop:-crop]
        annotators_samples = annotators_samples[:, crop:-crop]

        max_from_timesteps = [x.shape[-1] - (self.num_ahead + crop) for x in X_emb]
        from_timesteps = np.random.randint(
            low=min_from_timesteps,
            high=max_from_timesteps,
            size=batch_size,
        )

        contexts = self.apply_contexter(X_emb, device).transpose(2, 1)
        crop = Y_batch_samples.shape[-1] - contexts.shape[-2]
        valid_lengths -= crop
        if X_extra_batch_samples is not None:
            X_extra_batch_samples = X_extra_batch_samples[:, crop:]
        if Y_batch_samples is not None:
            Y_batch_samples = Y_batch_samples[:, crop:]
            annotators_samples = annotators_samples[:, crop:]

        if with_loss:
            batch_loss = self.cpc_loss(X_emb, contexts, from_timesteps, valid_lengths)
            return contexts, X_extra_batch_samples, Y_batch_samples, annotators_samples, batch_loss
        else:
            return contexts, X_extra_batch_samples, Y_batch_samples, annotators_samples


class MultiAnnotatorLogisticRegressionHead(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_annotators,
        num_extra_features=0,
        num_extra_clf_tasks=0,
        num_classes=4,
        num_extra_classes=2,
    ):
        super().__init__()

        annotator_embedding_size = num_annotators

        self.num_features_combined = num_features + num_extra_features

        self.ln = torch.nn.LayerNorm(num_features)

        self.logregs = [torch.nn.Linear(self.num_features_combined, num_classes)]
        for i in range(num_extra_clf_tasks):
            self.logregs.append(torch.nn.Linear(self.num_features_combined, num_extra_classes))
        self.logregs = torch.nn.ModuleList(self.logregs)

        self.residual = torch.nn.Sequential(
            torch.nn.Linear(
                self.num_features_combined + annotator_embedding_size,
                self.num_features_combined,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(
                self.num_features_combined,
                self.num_features_combined,
            ),
            torch.nn.LeakyReLU(),
        )

        self.embedding = torch.nn.Parameter((torch.diag(torch.ones(num_annotators))).detach())
        self.register_parameter(name="embedding", param=self.embedding)

        for m in self.modules():
            if isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, x_extra, annotators, clf_task):
        x = self.ln(x)
        a = self.embedding[annotators.flatten()].reshape(*annotators.shape, -1)

        x_ = torch.cat((x, x_extra, a), dim=-1)
        x_ = self.residual(x_)

        x = torch.cat((x, x_extra), dim=-1) + x_

        logits = self.logregs[clf_task](x)

        return logits
