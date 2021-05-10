import collections
import copy
import dataclasses
import pathlib

import madgrad
import numpy as np
import sklearn
import sklearn.metrics
import sklearn.utils
import torch
from fastprogress.fastprogress import progress_bar

import mabe.data
import mabe.loss
import mabe.model


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
    extra_task_loss_scaler: float = 0.1
    dark_annotator_loss_scaler: float = 0.2
    dark_knowledge_loss_scaler: float = 0.5
    fade_out_dark_knowledge: bool = False
    test_run: bool = False
    label_smoothing_task3: bool = True
    use_best_task0: bool = False


@dataclasses.dataclass
class TrainingResult:
    config: TrainingConfig
    losses: list
    clf_losses: dict
    clf_val_f1s: dict
    best_val_f1: dict
    best_params: dict
    final_params: tuple
    test_predictions: dict
    test_logits: dict
    task3_test_logits: dict
    params_by_epoch: list


class Trainer:
    config: TrainingConfig
    cpc: mabe.model.ConvCPC
    logreg: mabe.model.MultiAnnotatorLogisticRegressionHead
    scheduler: torch.optim.lr_scheduler._LRScheduler
    optimizer: torch.optim.Optimizer
    split: mabe.data.CVSplit
    data: mabe.data.DataWrapper
    device: str
    clf_loss: list
    dark_clf_loss: torch.nn.modules.loss._Loss

    losses: list
    clf_losses: dict
    dark_losses: dict
    clf_val_f1s: dict
    best_params: dict
    best_val_f1: dict
    params_by_epoch: list

    def __init__(self, config, data, split, device):
        self.config = config
        self.data = data
        self.split = split
        self.device = device

        if config.test_run:
            self.batches_per_epoch = 2
        else:
            self.batches_per_epoch = int(
                sum(data.sample_lengths) / config.subsample_length / config.batch_size
            )
        self.num_extra_clf_tasks = (
            len(np.unique(data.clf_tasks)) - 2
        )  # task12 clf and -1 for test data
        self.num_features = data.X[0].shape[-1]

        self.cpc = mabe.model.ConvCPC(
            self.num_features,
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
            num_extra_features=data.num_extra_features,
        ).to(device)

        self.logreg = mabe.model.MultiAnnotatorLogisticRegressionHead(
            config.num_context,
            data.num_annotators,
            data.num_extra_features,
            self.num_extra_clf_tasks,
        ).to(device)

        if config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                list(self.cpc.parameters()) + list(self.logreg.parameters()),
                weight_decay=config.weight_decay,
                lr=config.learning_rate,
                momentum=0.9,
                nesterov=True,
            )
        elif config.optimizer == "MADGRAD":
            self.optimizer = madgrad.MADGRAD(
                list(self.cpc.parameters()) + list(self.logreg.parameters()),
                weight_decay=config.weight_decay,
                lr=config.learning_rate,
            )

        self.clf_loss = []
        for task in range(self.num_extra_clf_tasks + 1):
            task_indices = np.argwhere(data.clf_tasks_labeled == task).flatten()
            task_train_Y = np.concatenate([data.Y_labeled[i] for i in task_indices]).astype(np.int)
            # TODO: class_weights only on train samples?
            class_weights = sklearn.utils.class_weight.compute_class_weight(
                "balanced", classes=np.unique(task_train_Y), y=task_train_Y
            )

            _, class_counts = np.unique(task_train_Y, return_counts=True)
            p_class = class_counts / np.sum(class_counts)

            if config.label_smoothing_task3:
                self.clf_loss.append(
                    mabe.loss.CrossEntropyLoss(
                        weight=torch.from_numpy(class_weights).to(device).float(),
                        ignore_index=-1,
                        smooth_eps=config.label_smoothing,
                        smooth_dist=torch.from_numpy(p_class).to(device).float(),
                    ).to(device)
                )
            else:
                self.clf_loss.append(
                    mabe.loss.CrossEntropyLoss(
                        weight=torch.from_numpy(class_weights).to(device).float(),
                        ignore_index=-1,
                    ).to(device)
                )

        # TODO: weight?
        self.dark_clf_loss = mabe.loss.CrossEntropyLoss().to(device)

        if config.scheduler == "cosine_annealing":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, config.num_epochs * self.batches_per_epoch
            )
        elif config.scheduler == "none":
            pass
        else:
            assert False

        self.losses = []
        self.clf_losses = collections.defaultdict(list)
        self.dark_losses = []
        self.clf_val_f1s = collections.defaultdict(list)
        self.best_params = {}
        self.best_val_f1 = {}
        self.best_val_f1_combined = 0.0
        self.params_by_epoch = []

        for task in range(self.num_extra_clf_tasks + 1):
            self.best_params[task] = (
                self.get_cpu_params(self.cpc),
                self.get_cpu_params(self.logreg),
            )

    def validation_f1(self, task: int):
        with torch.no_grad():
            cpc = self.cpc.eval()
            logreg = self.logreg.eval()

            predictions = []
            labels = []
            annotators = []

            crop_pre, crop_post = cpc.get_crops(self.device)

            def add_padding(seq):
                return np.concatenate(
                    (
                        np.zeros_like(seq)[:crop_pre],
                        seq,
                        np.zeros_like(seq)[:crop_post],
                    )
                )

            with torch.no_grad():
                for idx in self.split.val_indices_labeled:
                    if self.data.clf_tasks[idx] != task:
                        continue

                    y = self.data.Y_labeled[idx]
                    a = np.array([self.data.annotators_labeled[idx]]).repeat(len(y))
                    if task > 0:
                        assert np.all(a == 0)  # only first annotator for task 3
                    x = add_padding(self.data.X_labeled[idx].astype(np.float32))
                    if self.config.use_extra_features:
                        x_extra = self.data.X_labeled_extra[idx].astype(np.float32)
                        x_extra = torch.from_numpy(x_extra).to(self.device, non_blocking=True)

                    x = torch.transpose(torch.from_numpy(x[None, :, :]), 2, 1).to(
                        self.device, non_blocking=True
                    )
                    x_emb = cpc.embedder(x)

                    c = cpc.apply_contexter(x_emb, self.device)

                    logreg_features = c[0].T
                    l = logreg(logreg_features, x_extra, a, task)
                    p = torch.argmax(l, dim=-1)

                    predictions.append(p.cpu().numpy())
                    labels.append(y)
                    annotators.append(a)

        if len(predictions):
            annotators = np.concatenate(annotators).astype(np.int)
            predictions = np.concatenate(predictions).astype(np.int)
            labels_array = np.concatenate(labels).astype(np.int)

            if task == 0:
                # validation loss only for first annotator
                predictions = predictions[annotators == 0]
                labels_array = labels_array[annotators == 0]

                return mabe.loss.macro_f1_score(labels_array, predictions, 4)
            else:
                assert labels_array.max() == 1
                assert labels_array.min() == 0
                # calculate F1 score for behavior, not macro F1
                # return mabe.loss.f1_score_for_label(labels_array, predictions, 2, 1)
                return sklearn.metrics.f1_score(labels_array, predictions)
        else:
            return None

    def clf_task_loss(
        self,
        batch,
        X_extra_batch,
        Y_batch,
        contexts,
        annotators_batch,
        task,
    ):
        has_train_labels = batch.clf_tasks == task

        Y_batch_flat = Y_batch[has_train_labels].flatten().long()
        valids = Y_batch_flat >= 0
        assert torch.any(valids)

        logreg_features = contexts[has_train_labels]
        annotators_batch = annotators_batch[has_train_labels]
        if self.config.use_extra_features:
            X_extra_batch = X_extra_batch[has_train_labels]

        num_classes = 4 if task == 0 else 2
        clf_batch_loss = self.clf_loss[task](
            self.logreg(logreg_features, X_extra_batch, annotators_batch, task).reshape(
                -1, num_classes
            ),
            Y_batch_flat,
        )

        assert np.all(
            (annotators_batch.flatten() >= 0)
            | ((annotators_batch.flatten() == -1) & (Y_batch_flat.cpu().data.numpy() == -1))
        )

        return clf_batch_loss

    def dark_clf_losses(
        self,
        contexts,
        X_extra_batch,
        Y_batch_dark_behaviors,
        Y_batch_dark_annotators,
        annotators_batch,
    ):
        has_dark_labels = (Y_batch_dark_behaviors >= 0.0).sum(dim=(1, 2, 3)) > 0
        logreg_features = contexts[has_dark_labels]
        annotators_batch = np.zeros_like(annotators_batch[has_dark_labels.cpu().data.numpy()])
        if self.config.use_extra_features:
            X_extra_batch = X_extra_batch[has_dark_labels]

        dark_behavior_losses_batch = []
        num_classes = 2
        for task in range(1, self.num_extra_clf_tasks + 1):
            behavior = task - 1

            behavior_logits = Y_batch_dark_behaviors[has_dark_labels][:, :, behavior].reshape(
                -1, num_classes
            )
            dark_clf_batch_loss = self.dark_clf_loss(
                self.logreg(logreg_features, X_extra_batch, annotators_batch, task).reshape(
                    -1, num_classes
                ),
                behavior_logits,
            )
            dark_behavior_losses_batch.append(dark_clf_batch_loss)
        dark_behavior_loss_batch = (
            sum(dark_behavior_losses_batch) * self.config.extra_task_loss_scaler
        )

        dark_annotator_losses_batch = []
        num_classes = 4
        task = 0
        sum_annotators = 0
        for annotator in range(self.data.num_annotators):
            annotator_logits = Y_batch_dark_annotators[has_dark_labels][:, :, annotator].reshape(
                -1, num_classes
            )
            dark_clf_batch_loss = self.dark_clf_loss(
                self.logreg(logreg_features, X_extra_batch, annotators_batch, task).reshape(
                    -1, num_classes
                ),
                annotator_logits,
            )

            scaler = 1 if annotator == 0 else self.config.dark_annotator_loss_scaler
            dark_clf_batch_loss *= scaler
            sum_annotators += scaler

            dark_annotator_losses_batch.append(dark_clf_batch_loss)
        dark_annotator_loss_batch = sum(dark_annotator_losses_batch) / sum_annotators

        return dark_behavior_loss_batch + dark_annotator_loss_batch

    def train_batch(self, epoch):
        self.optimizer.zero_grad()
        cpc = self.cpc.train()

        batch = self.split.get_train_batch(
            self.config.batch_size,
            random_noise=self.config.augmentation_random_noise,
            extra_features=self.config.use_extra_features,
            dark_knowledge=self.config.dark_knowledge_loss_scaler > 0.0,
        )

        (
            contexts,
            X_extra_batch,
            Y_batch,
            Y_batch_dark_behaviors,
            Y_batch_dark_annotators,
            annotators_batch,
            batch_loss,
        ) = cpc(batch, device=self.device, with_loss=True)
        self.losses.append(batch_loss.cpu().item())

        batch_clf_task_losses = []
        for task in range(self.num_extra_clf_tasks + 1):
            task_loss = self.clf_task_loss(
                batch,
                X_extra_batch,
                Y_batch,
                contexts,
                annotators_batch,
                task,
            )

            self.clf_losses[task].append(task_loss.item())

            loss_scaler = 1
            if task > 0:
                loss_scaler = self.config.extra_task_loss_scaler

            task_loss *= loss_scaler
            batch_clf_task_losses.append(task_loss)
        batch_clf_task_loss = sum(batch_clf_task_losses)

        if self.config.dark_knowledge_loss_scaler > 0.0:
            dark_knowledge_loss = self.dark_clf_losses(
                contexts,
                X_extra_batch,
                Y_batch_dark_behaviors,
                Y_batch_dark_annotators,
                annotators_batch,
            )
            self.dark_losses.append(dark_knowledge_loss.item())
        else:
            dark_knowledge_loss = 0.0

        epoch_scaler = 1.0
        if self.config.fade_out_dark_knowledge:
            epoch_scaler = (self.config.num_epochs / (epoch + 1)) / self.config.num_epochs

        batch_loss = (
            batch_loss
            + self.config.clf_loss_scaling * batch_clf_task_loss
            + self.config.dark_knowledge_loss_scaler * dark_knowledge_loss * epoch_scaler
        )

        batch_loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def running(self, losses):
        return np.mean([i for i in losses[-self.batches_per_epoch :] if i is not None])

    def log_batch(self, bar):
        task3_running_clf_loss = []
        for task in range(1, self.num_extra_clf_tasks + 1):
            task3_running_clf_loss.append(self.running(self.clf_losses[task]))
        task3_running_clf_loss = np.mean(task3_running_clf_loss)

        task3_running_val_f1 = []
        for task in range(1, self.num_extra_clf_tasks + 1):
            task3_running_val_f1.append(self.best_val_f1[task])
        task3_running_val_f1 = np.mean([i for i in task3_running_val_f1 if i is not None])

        bar.comment = (
            f"Train: {self.running(self.losses):.3f} | "
            + f"CLF Train: {self.running(self.clf_losses[0]):.3f} | "
            + f"CLF Train [T3]: {task3_running_clf_loss:.3f} | "
            + f"Dark: {self.running(self.dark_losses):.3f} | "
            + f"Best CLF Val F1: {self.best_val_f1[0]:.3f} | "
            + f"Best CLF Val F1 [T3]: {task3_running_val_f1:.3f} | "
            + f"LR: {self.get_lr(self.optimizer):.4f}"
        )

    @staticmethod
    def get_cpu_params(model):
        return copy.deepcopy({k: v.cpu().detach() for k, v in model.state_dict().items()})

    def finalize_epoch(self):
        for task in range(self.num_extra_clf_tasks + 1):
            val_f1 = self.validation_f1(task)

            if val_f1 is not None:
                if val_f1 > self.best_val_f1[task]:
                    self.best_params[task] = (
                        self.get_cpu_params(self.cpc),
                        self.get_cpu_params(self.logreg),
                    )
                self.best_val_f1[task] = max(val_f1, self.best_val_f1[task])
            self.clf_val_f1s[task].append(val_f1)

        # mean of validation f1s for all task3 subtasks
        """
        val_f1 = np.mean(
            [
                self.clf_val_f1s[task][-1]
                for task in range(1, self.num_extra_clf_tasks + 1)
                if self.clf_val_f1s[task][-1] is not None
            ]
        )
        if val_f1 > self.best_val_f1_combined:
            # no validation data for task 3.3, use mean of other task 3 subtasks
            task = 4
            self.best_params[task] = (
                self.get_cpu_params(self.cpc),
                self.get_cpu_params(self.logreg),
            )
            self.best_val_f1_combined = val_f1
        """

        self.params_by_epoch.append(
            (
                self.get_cpu_params(self.cpc),
                self.get_cpu_params(self.logreg),
            )
        )

    def get_result(self) -> TrainingResult:
        final_params = (
            self.get_cpu_params(self.cpc),
            self.get_cpu_params(self.logreg),
        )

        test_predictions, test_logits, task3_test_logits = mabe.util.predict_test_data(
            self.cpc, self.logreg, self.data, self.device, self.config, self.best_params
        )

        result = TrainingResult(
            config=self.config,
            losses=self.losses,
            clf_losses=self.clf_losses,
            clf_val_f1s=self.clf_val_f1s,
            best_val_f1=self.best_val_f1,
            best_params=self.best_params,
            params_by_epoch=self.params_by_epoch,
            final_params=final_params,
            test_predictions=test_predictions,
            test_logits=test_logits,
            task3_test_logits=task3_test_logits,
        )

        return result

    def train_model(self) -> TrainingResult:
        for task in range(self.num_extra_clf_tasks + 1):
            val_f1 = self.validation_f1(task)
            self.best_val_f1[task] = val_f1
            self.clf_val_f1s[task].append(val_f1)

        # use combined bar for epochs and batches
        bar = progress_bar(range(self.config.num_epochs * self.batches_per_epoch))
        bar_iter = iter(bar)
        for i_epoch in range(self.config.num_epochs):
            for _ in range(self.batches_per_epoch):
                next(bar_iter)
                self.train_batch(i_epoch)
                self.log_batch(bar)

            self.finalize_epoch()

        return self.get_result()
