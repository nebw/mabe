import collections
import io

import numpy as np
import torch
from fastprogress.fastprogress import force_console_behavior

master_bar, progress_bar = force_console_behavior()


def predict_test_data(
    cpc, logreg, data, device, config, params, fixed_params=False, task12=True, task3=True
):
    def load_task_params(task):
        if config.use_best_task0:
            task = 0

        if fixed_params:
            cpc_state_dict, logreg_state_dict = params
        else:
            cpc_state_dict, logreg_state_dict = params[task]
        cpc.load_state_dict(cpc_state_dict)
        logreg.load_state_dict(logreg_state_dict)
        cpc_ = cpc.eval()
        logreg_ = logreg.eval()
        return cpc_, logreg_

    crop_pre, crop_post = cpc.get_crops(device)

    def add_padding(seq):
        return np.concatenate(
            (
                np.zeros_like(seq)[:crop_pre],
                seq,
                np.zeros_like(seq)[:crop_post],
            )
        )

    cpc, logreg = load_task_params(0)
    test_predictions = collections.defaultdict(dict)
    test_logits = collections.defaultdict(dict)
    if task12:
        with torch.no_grad():
            bar = progress_bar(range(len(data.X_unlabeled)))
            for idx in bar:
                x = add_padding(data.X_unlabeled[idx].astype(np.float32))
                x_extra = None
                if config.use_extra_features:
                    x_extra = data.X_unlabeled_extra[idx].astype(np.float32)
                    x_extra = torch.from_numpy(x_extra).to(device, non_blocking=True)

                g = data.groups_unlabeled[idx]

                x = torch.transpose(torch.from_numpy(x[None, :, :]), 2, 1).to(
                    device, non_blocking=True
                )
                x_emb = cpc.embedder(x)

                c = cpc.apply_contexter(x_emb, device)

                logreg_features = c[0].T

                for annotator in range(data.num_annotators):
                    a = np.array([annotator]).repeat(len(logreg_features))
                    task = 0
                    l = logreg(logreg_features, x_extra, a, task)

                    l = torch.cat((l[:1], l), dim=0)  # crop from feature preprocessing
                    p = torch.argmax(l, dim=-1)

                    assert len(p) == len(data.X_unlabeled[idx]) + 1

                    with io.BytesIO() as buffer:
                        np.savez_compressed(buffer, p.cpu().data.numpy())
                        test_predictions[g.decode("utf-8")][annotator] = buffer.getvalue()
                    with io.BytesIO() as buffer:
                        np.savez_compressed(buffer, l.cpu().data.numpy().astype(np.float32))
                        test_logits[g.decode("utf-8")][annotator] = buffer.getvalue()

    task3_test_logits = collections.defaultdict(dict)
    if task3:
        annotator = 0  # only first annotator for task 3
        for task in range(1, max(data.clf_tasks) + 1):
            cpc, logreg = load_task_params(task)

            with torch.no_grad():
                # bar = progress_bar(range(len(data.X_unlabeled)))
                bar = range(len(data.X_unlabeled))
                for idx in bar:
                    x = add_padding(data.X_unlabeled[idx].astype(np.float32))
                    x_extra = None
                    if config.use_extra_features:
                        x_extra = data.X_unlabeled_extra[idx].astype(np.float32)
                        x_extra = torch.from_numpy(x_extra).to(device, non_blocking=True)

                    g = data.groups_unlabeled[idx]

                    x = torch.transpose(torch.from_numpy(x[None, :, :]), 2, 1).to(
                        device, non_blocking=True
                    )
                    x_emb = cpc.embedder(x)

                    c = cpc.apply_contexter(x_emb, device)

                    logreg_features = c[0].T
                    a = np.array([annotator]).repeat(len(logreg_features))
                    l = logreg(logreg_features, x_extra, a, task)

                    l = torch.cat((l[:1], l), dim=0)  # crop from feature preprocessing
                    p = torch.argmax(l, dim=-1)

                    assert len(p) == len(data.X_unlabeled[idx]) + 1

                    with io.BytesIO() as buffer:
                        np.savez_compressed(buffer, l.cpu().data.numpy().astype(np.float32))
                        task3_test_logits[g.decode("utf-8")][task] = buffer.getvalue()

    return test_predictions, test_logits, task3_test_logits
