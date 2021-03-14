import numpy as np
import torch
from fastprogress.fastprogress import force_console_behavior

import mabe
import mabe.custom_lstms

master_bar, progress_bar = force_console_behavior()


class CausalConv1D(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        downsample=False,
        layer_norm=False,
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
        self.layer_norm = layer_norm

    def forward(self, x):
        if self.layer_norm:
            x = torch.nn.functional.layer_norm(x, x.shape[1:])

        x = self.conv(x)

        # remove trailing padding
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
                CausalConv1D(out_channels, out_channels, dilation=2, layer_norm=True),
                CausalConv1D(out_channels, out_channels, dilation=2, layer_norm=True),
                CausalConv1D(out_channels, out_channels, dilation=2, layer_norm=True),
                CausalConv1D(out_channels, out_channels, dilation=2, layer_norm=True),
                CausalConv1D(out_channels, out_channels, dilation=2, layer_norm=True),
                CausalConv1D(out_channels, out_channels, layer_norm=True),
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


class CPC(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_embeddings,
        num_context,
        num_ahead,
        num_ahead_subsampling,
        subsample_length,
        num_rnn_layers=2,
        dropout=0,
        reverse_time=False,
        **kwargs,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.num_context = num_context
        self.num_rnn_layers = num_rnn_layers

        self.embedder = Embedder(num_features, num_embeddings, dropout=dropout)
        self.contexter = mabe.custom_lstms.script_lnlstm(
            num_embeddings, num_context, num_rnn_layers, batch_first=False
        )

        self.projections = torch.nn.ModuleList(
            [
                torch.nn.Linear(num_context, num_embeddings, bias=False)
                for _ in range(num_ahead // num_ahead_subsampling)
            ]
        )

        self.num_ahead = num_ahead
        self.num_ahead_subsampling = num_ahead_subsampling
        self.subsample_length = subsample_length
        self.reverse_time = reverse_time

    def subsample_and_pad(self, X, Y=None, device="cpu"):
        batch_size = len(X)

        padding = np.array([max(0, self.subsample_length + self.num_ahead - len(x)) for x in X])

        X = [np.pad(x, ((0, pad), (0, 0))) for x, pad in zip(X, padding)]

        valid_lengths = np.array([len(x) - pad for x, pad in zip(X, padding)])
        subsample_from_max = np.clip(
            valid_lengths - self.subsample_length - self.num_ahead, 0, np.inf
        )
        subsample_from = np.floor(np.random.rand(batch_size) * subsample_from_max).astype(np.int)

        X = np.stack(
            [
                X[i][subsample_from[i] : subsample_from[i] + self.subsample_length + self.num_ahead]
                for i in range(batch_size)
            ]
        )

        X = torch.transpose(torch.from_numpy(X), 2, 1).to(device)

        if Y is not None:
            Y = [np.pad(y, ((0, pad)), constant_values=-1.0) for y, pad in zip(Y, padding)]

            Y = np.stack(
                [
                    Y[i][
                        subsample_from[i] : subsample_from[i]
                        + self.subsample_length
                        + self.num_ahead
                    ]
                    for i in range(batch_size)
                ]
            )
            Y = torch.from_numpy(Y).to(device)

        return X, Y, valid_lengths

    def cpc_loss(self, X_emb, contexts, from_timesteps, valid_lengths):
        batch_size = len(contexts)

        from_timesteps_reduced = from_timesteps
        contexts_from = torch.stack(
            [contexts[i, from_timesteps_reduced[i]] for i in range(batch_size)]
        )

        ahead = 0
        batch_loss = []
        for ahead in range(1, self.num_ahead, self.num_ahead_subsampling):
            embeddings_projected = self.projections[ahead // self.num_ahead_subsampling](
                contexts_from
            )
            embeddings_projected = torch.nn.functional.normalize(embeddings_projected, p=2, dim=-1)
            embeddings_to = torch.stack(
                [X_emb[i, :, from_timesteps_reduced[i] + ahead] for i in range(batch_size)]
            )
            embeddings_to = torch.nn.functional.normalize(embeddings_to, p=2, dim=-1)

            # assume both are l2-normalized -> cosine similarity
            predictions_to = torch.einsum("ae,ce->ac", embeddings_projected, embeddings_to)

            has_value_to = torch.from_numpy((from_timesteps_reduced + ahead) < valid_lengths).to(
                X_emb.device
            )
            loss = nt_xent_loss(predictions_to, has_value_to)
            loss_mean = torch.mean(loss)

            batch_loss.append(loss_mean)

        batch_loss = sum(batch_loss) / len(batch_loss)

        return batch_loss

    def get_contexts(self, X, device="cpu"):
        embedder = self.embedder.eval()
        contexter = self.contexter.eval()

        contexts = []
        with torch.no_grad():
            bar = progress_bar(range(len(X)))
            for idx in bar:
                x = X[idx].astype(np.float32)
                if self.reverse_time:
                    x = x[::-1]

                x = torch.transpose(torch.from_numpy(x[None, :, :]), 2, 1).to(device)
                x_emb = embedder(x)

                c, _ = contexter(x_emb.transpose(2, 1))

                contexts.append(c.cpu().numpy()[0])

        contexts = np.concatenate(contexts)
        return contexts

    def forward(self, X_batch, Y_batch=None, device="cpu", with_loss=False):
        batch_size = len(X_batch)

        if self.reverse_time:
            X_batch = [x[::-1] for x in X_batch]

            if Y_batch is not None:
                Y_batch = [y[::-1] for y in Y_batch]

        X_batch, Y_batch, valid_lengths = self.subsample_and_pad(X_batch, Y_batch, device)

        X_emb = self.embedder(X_batch)

        max_from_timesteps = [x.shape[-1] - self.num_ahead for x in X_emb]
        from_timesteps = np.random.randint(
            low=0,
            high=max_from_timesteps,
            size=batch_size,
        )
        # contexts, _ = self.contexter(X_emb[:, :, : max(from_timesteps) + 1].transpose(2, 1))
        rnn_input = X_emb.permute(2, 0, 1)
        states = [
            mabe.custom_lstms.LSTMState(
                torch.randn(batch_size, self.num_context, device=device),
                torch.randn(batch_size, self.num_context, device=device),
            )
            for _ in range(self.num_rnn_layers)
        ]
        contexts, _ = self.contexter(rnn_input, states)
        contexts = contexts.transpose(1, 0)

        if with_loss:
            batch_loss = self.cpc_loss(X_emb, contexts, from_timesteps, valid_lengths)
            return contexts, Y_batch, batch_loss
        else:
            return contexts, Y_batch
