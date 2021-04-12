import math

import numba
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output


def _is_long(x):
    if hasattr(x, "data"):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


def cross_entropy(
    inputs,
    target,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    smooth_eps=None,
    smooth_dist=None,
    from_logits=True,
):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target) and smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(
                inputs, target, weight, ignore_index=ignore_index, reduction=reduction
            )
        else:
            return F.nll_loss(
                inputs, target, weight, ignore_index=ignore_index, reduction=reduction
            )

    if from_logits:
        # log-softmax of inputs
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = inputs

    masked_indices = None
    num_classes = inputs.size(-1)

    if _is_long(target):
        masked_indices = target.eq(ignore_index)
        lsm = lsm[~masked_indices]
        target = target[~masked_indices]

    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1.0 - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()

    return loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(
        self,
        weight=None,
        ignore_index=-100,
        reduction="mean",
        smooth_eps=None,
        smooth_dist=None,
        from_logits=True,
    ):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, input, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        return cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            smooth_eps=self.smooth_eps,
            smooth_dist=smooth_dist,
            from_logits=self.from_logits,
        )


def binary_cross_entropy(
    inputs, target, weight=None, reduction="mean", smooth_eps=None, from_logits=False
):
    """cross entropy loss, with support for label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0
    if smooth_eps > 0:
        target = target.float()
        target.add_(smooth_eps).div_(2.0)
    if from_logits:
        return F.binary_cross_entropy_with_logits(
            inputs, target, weight=weight, reduction=reduction
        )
    else:
        return F.binary_cross_entropy(inputs, target, weight=weight, reduction=reduction)


def binary_cross_entropy_with_logits(
    inputs, target, weight=None, reduction="mean", smooth_eps=None, from_logits=True
):
    return binary_cross_entropy(inputs, target, weight, reduction, smooth_eps, from_logits)


class BCELoss(nn.BCELoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        smooth_eps=None,
        from_logits=False,
    ):
        super().__init__(weight, size_average, reduce, reduction)
        self.smooth_eps = smooth_eps
        self.from_logits = from_logits

    def forward(self, input, target):
        return binary_cross_entropy(
            input,
            target,
            weight=self.weight,
            reduction=self.reduction,
            smooth_eps=self.smooth_eps,
            from_logits=self.from_logits,
        )


class BCEWithLogitsLoss(BCELoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        smooth_eps=None,
        from_logits=True,
    ):
        super().__init__(
            weight, size_average, reduce, reduction, smooth_eps=smooth_eps, from_logits=from_logits
        )


def range_to_anchors_and_delta(precision_range, num_anchors):
    """Calculates anchor points from precision range.
    Args:
        precision_range: an interval (a, b), where 0.0 <= a <= b <= 1.0
        num_anchors: int, number of equally spaced anchor points.
    Returns:
        precision_values: A `Tensor` of [num_anchors] equally spaced values
            in the interval precision_range.
        delta: The spacing between the values in precision_values.
    Raises:
        ValueError: If precision_range is invalid.
    """
    # Validate precision_range.
    if len(precision_range) != 2:
        raise ValueError("length of precision_range (%d) must be 2" % len(precision_range))
    if not 0 <= precision_range[0] <= precision_range[1] <= 1:
        raise ValueError(
            "precision values must follow 0 <= %f <= %f <= 1"
            % (precision_range[0], precision_range[1])
        )

    # Sets precision_values uniformly between min_precision and max_precision.
    precision_values = numpy.linspace(
        start=precision_range[0], stop=precision_range[1], num=num_anchors + 1
    )[1:]

    delta = (precision_range[1] - precision_range[0]) / num_anchors
    return FloatTensor(precision_values), delta


def build_class_priors(
    labels,
    class_priors=None,
    weights=None,
    positive_pseudocount=1.0,
    negative_pseudocount=1.0,
):
    """build class priors, if necessary.
    For each class, the class priors are estimated as
    (P + sum_i w_i y_i) / (P + N + sum_i w_i),
    where y_i is the ith label, w_i is the ith weight, P is a pseudo-count of
    positive labels, and N is a pseudo-count of negative labels.
    Args:
        labels: A `Tensor` with shape [batch_size, num_classes].
            Entries should be in [0, 1].
        class_priors: None, or a floating point `Tensor` of shape [C]
            containing the prior probability of each class (i.e. the fraction of the
            training data consisting of positive examples). If None, the class
            priors are computed from `targets` with a moving average.
        weights: `Tensor` of shape broadcastable to labels, [N, 1] or [N, C],
            where `N = batch_size`, C = num_classes`
        positive_pseudocount: Number of positive labels used to initialize the class
            priors.
        negative_pseudocount: Number of negative labels used to initialize the class
            priors.
    Returns:
        class_priors: A Tensor of shape [num_classes] consisting of the
          weighted class priors, after updating with moving average ops if created.
    """
    if class_priors is not None:
        return class_priors

    N, C = labels.size()

    weighted_label_counts = (weights * labels).sum(0)

    weight_sum = weights.sum(0)

    class_priors = torch.div(
        weighted_label_counts + positive_pseudocount,
        weight_sum + positive_pseudocount + negative_pseudocount,
    )

    return class_priors


def weighted_hinge_loss(labels, logits, positive_weights=1.0, negative_weights=1.0):
    """
    Args:
        labels: one-hot representation `Tensor` of shape broadcastable to logits
        logits: A `Tensor` of shape [N, C] or [N, C, K]
        positive_weights: Scalar or Tensor
        negative_weights: same shape as positive_weights
    Returns:
        3D Tensor of shape [N, C, K], where K is length of positive weights
        or 2D Tensor of shape [N, C]
    """
    positive_weights_is_tensor = torch.is_tensor(positive_weights)
    negative_weights_is_tensor = torch.is_tensor(negative_weights)

    # Validate positive_weights and negative_weights
    if positive_weights_is_tensor ^ negative_weights_is_tensor:
        raise ValueError(
            "positive_weights and negative_weights must be same shape Tensor "
            "or both be scalars. But positive_weight_is_tensor: %r, while "
            "negative_weight_is_tensor: %r"
            % (positive_weights_is_tensor, negative_weights_is_tensor)
        )

    if positive_weights_is_tensor and (positive_weights.size() != negative_weights.size()):
        raise ValueError(
            "shape of positive_weights and negative_weights "
            "must be the same! "
            "shape of positive_weights is {0}, "
            "but shape of negative_weights is {1}"
            % (positive_weights.size(), negative_weights.size())
        )

    # positive_term: Tensor [N, C] or [N, C, K]
    positive_term = (1 - logits).clamp(min=0) * labels
    negative_term = (1 + logits).clamp(min=0) * (1 - labels)

    if positive_weights_is_tensor and positive_term.dim() == 2:
        return (
            positive_term.unsqueeze(-1) * positive_weights
            + negative_term.unsqueeze(-1) * negative_weights
        )
    else:
        return positive_term * positive_weights + negative_term * negative_weights


def true_positives_lower_bound(labels, logits, weights):
    """
    true_positives_lower_bound defined in paper:
    "Scalable Learning of Non-Decomposable Objectives"
    Args:
        labels: A `Tensor` of shape broadcastable to logits.
        logits: A `Tensor` of shape [N, C] or [N, C, K].
            If the third dimension is present,
            the lower bound is computed on each slice [:, :, k] independently.
        weights: Per-example loss coefficients, with shape [N, 1] or [N, C]
    Returns:
        A `Tensor` of shape [C] or [C, K].
    """
    # A `Tensor` of shape [N, C] or [N, C, K]
    loss_on_positives = weighted_hinge_loss(labels, logits, negative_weights=0.0)

    weighted_loss_on_positives = (
        weights.unsqueeze(-1) * (labels - loss_on_positives)
        if loss_on_positives.dim() > weights.dim()
        else weights * (labels - loss_on_positives)
    )
    return weighted_loss_on_positives.sum(0)


def false_postives_upper_bound(labels, logits, weights):
    """
    false_positives_upper_bound defined in paper:
    "Scalable Learning of Non-Decomposable Objectives"
    Args:
        labels: A `Tensor` of shape broadcastable to logits.
        logits: A `Tensor` of shape [N, C] or [N, C, K].
            If the third dimension is present,
            the lower bound is computed on each slice [:, :, k] independently.
        weights: Per-example loss coefficients, with shape broadcast-compatible with
            that of `labels`. i.e. [N, 1] or [N, C]
    Returns:
        A `Tensor` of shape [C] or [C, K].
    """
    loss_on_negatives = weighted_hinge_loss(labels, logits, positive_weights=0)

    weighted_loss_on_negatives = (
        weights.unsqueeze(-1) * loss_on_negatives
        if loss_on_negatives.dim() > weights.dim()
        else weights * loss_on_negatives
    )
    return weighted_loss_on_negatives.sum(0)


class LagrangeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def lagrange_multiplier(x):
    return LagrangeMultiplier.apply(x)


class AUCPRHingeLoss(nn.Module):
    """area under the precision-recall curve loss,
    Reference: "Scalable Learning of Non-Decomposable Objectives", Section 5 \
    TensorFlow Implementation: \
    https://github.com/tensorflow/models/tree/master/research/global_objectives\
    """

    class Config:
        r"""
        Attributes:
            precision_range_lower (float): the lower range of precision values over
                which to compute AUC. Must be nonnegative, `\leq precision_range_upper`,
                and `leq 1.0`.
            precision_range_upper (float): the upper range of precision values over
                which to compute AUC. Must be nonnegative, `\geq precision_range_lower`,
                and `leq 1.0`.
            num_classes (int): number of classes(aka labels)
            num_anchors (int): The number of grid points used to approximate the
                Riemann sum.
        """

        precision_range_lower: float = 0.0
        precision_range_upper: float = 1.0
        num_classes: int = 1
        num_anchors: int = 20

    def __init__(self, config, weights=None, *args, **kwargs):
        """Args:
        config: Config containing `precision_range_lower`, `precision_range_upper`,
            `num_classes`, `num_anchors`
        """
        nn.Module.__init__(self)
        self.config = config

        self.num_classes = self.config.num_classes
        self.num_anchors = self.config.num_anchors
        self.precision_range = (
            self.config.precision_range_lower,
            self.config.precision_range_upper,
        )

        # Create precision anchor values and distance between anchors.
        # coresponding to [alpha_t] and [delta_t] in the paper.
        # precision_values: 1D `Tensor` of shape [K], where `K = num_anchors`
        # delta: Scalar (since we use equal distance between anchors)
        self.precision_values, self.delta = range_to_anchors_and_delta(
            self.precision_range, self.num_anchors
        )

        # notation is [b_k] in paper, Parameter of shape [C, K]
        # where `C = number of classes` `K = num_anchors`
        self.biases = nn.Parameter(
            FloatTensor(self.config.num_classes, self.config.num_anchors).zero_()
        )
        self.lambdas = nn.Parameter(
            FloatTensor(self.config.num_classes, self.config.num_anchors).data.fill_(1.0)
        )

    def forward(self, logits, targets, reduce=True, size_average=True, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
            size_average (bool, optional): By default, the losses are averaged
                    over observations for each minibatch. However, if the field
                    sizeAverage is set to False, the losses are instead summed
                    for each minibatch. Default: ``True``
            reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on size_average. When reduce
                is False, returns a loss per input/target element instead and ignores
                size_average. Default: True
        """
        C = 1 if logits.dim() == 1 else logits.size(1)

        if self.num_classes != C:
            raise ValueError("num classes is %d while logits width is %d" % (self.num_classes, C))

        labels, weights = AUCPRHingeLoss._prepare_labels_weights(logits, targets, weights=weights)

        # Lagrange multipliers
        # Lagrange multipliers are required to be nonnegative.
        # Their gradient is reversed so that they are maximized
        # (rather than minimized) by the optimizer.
        # 1D `Tensor` of shape [K], where `K = num_anchors`
        lambdas = lagrange_multiplier(self.lambdas)
        # print("lambdas: {}".format(lambdas))

        precision_values = self.precision_values.to(logits.device)

        # A `Tensor` of Shape [N, C, K]
        hinge_loss = weighted_hinge_loss(
            labels.unsqueeze(-1),
            logits.unsqueeze(-1) - self.biases,
            positive_weights=1.0 + lambdas * (1.0 - precision_values),
            negative_weights=lambdas * precision_values,
        )

        # 1D tensor of shape [C]
        class_priors = build_class_priors(labels, weights=weights)

        # lambda_term: Tensor[C, K]
        # according to paper, lambda_term = lambda * (1 - precision) * |Y^+|
        # where |Y^+| is number of postive examples = N * class_priors
        lambda_term = class_priors.unsqueeze(-1) * (lambdas * (1.0 - precision_values))

        per_anchor_loss = weights.unsqueeze(-1) * hinge_loss - lambda_term

        # Riemann sum over anchors, and normalized by precision range
        # loss: Tensor[N, C]
        loss = per_anchor_loss.sum(2) * self.delta
        loss /= self.precision_range[1] - self.precision_range[0]

        if not reduce:
            return loss
        elif size_average:
            return loss.mean()
        else:
            return loss.sum()

    @staticmethod
    def _prepare_labels_weights(logits, targets, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
        Returns:
            labels: Tensor of shape [N, C], one-hot representation
            weights: Tensor of shape broadcastable to labels
        """
        N, C = logits.size()
        # Converts targets to one-hot representation. Dim: [N, C]
        labels = (
            FloatTensor(N, C).to(logits.device).zero_().scatter(1, targets.unsqueeze(1).data, 1)
        )

        if weights is None:
            weights = FloatTensor(N).to(logits.device).fill_(1.0)

        if weights.dim() == 1:
            weights.unsqueeze_(-1)

        weights = weights.detach()

        return labels, weights


def f1_loss(Y_batch_flat, Y_pred_flat, valids):
    valids = (Y_batch_flat >= 0).cpu().numpy()
    y_true = Y_batch_flat[valids]
    y_pred = torch.nn.functional.softmax(Y_pred_flat[valids], dim=-1)
    y_true = onehot(y_true, N=4)
    eps = 1e-8

    tp = (y_true * y_pred).sum(dim=0)
    # tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0)
    fp = ((1 - y_true) * y_pred).sum(dim=0)
    fn = (y_true * (1 - y_pred)).sum(dim=0)

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    f1 = 2 * p * r / (p + r + eps)

    return 1 - f1.mean()


@numba.njit
def macro_f1_score(y_true, y_pred, n_labels):
    total_f1 = 0.0
    for i in range(n_labels):
        yt = y_true == i
        yp = y_pred == i

        tp = np.sum(yt & yp)

        tpfp = np.sum(yp)
        tpfn = np.sum(yt)
        if tpfp == 0:
            print(
                "[WARNING] F-score is ill-defined and being set to 0.0 in labels with no predicted samples."
            )
            precision = 0.0
        else:
            precision = tp / tpfp
        if tpfn == 0:
            print(f"[ERROR] label not found in y_true...")
            recall = 0.0
        else:
            recall = tp / tpfn

        if precision == 0.0 or recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        total_f1 += f1
    return total_f1 / n_labels


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
