import collections
import io

import numpy as np
import torch
from fastprogress.fastprogress import progress_bar


def predict_test_data(cpc, logreg, data, device, config, params):
    cpc_state_dict, logreg_state_dict = params
    cpc.load_state_dict(cpc_state_dict)
    logreg.load_state_dict(logreg_state_dict)

    cpc = cpc.eval()
    logreg = logreg.eval()

    test_predictions = collections.defaultdict(dict)
    test_logits = collections.defaultdict(dict)
    task3_test_logits = collections.defaultdict(dict)
    with torch.no_grad():
        bar = progress_bar(range(len(data.X_unlabeled)))
        for idx in bar:
            # from feature preprocessing
            crop_pre = 1
            crop_post = 0

            x = data.X_unlabeled[idx].astype(np.float32)
            x_extra = None
            if config.use_extra_features:
                x_extra = data.X_unlabeled_extra[idx].astype(np.float32)
                x_extra = torch.from_numpy(x_extra).to(device, non_blocking=True)

            g = data.groups_unlabeled[idx]

            x = torch.transpose(torch.from_numpy(x[None, :, :]), 2, 1).to(device, non_blocking=True)
            x_emb = cpc.embedder(x)

            crop = (x.shape[-1] - x_emb.shape[-1]) // 2
            crop_pre += crop
            crop_post += crop

            c = cpc.apply_contexter(x_emb, device)

            crop = x_emb.shape[-1] - c.shape[-1]
            crop_pre += crop

            logreg_features = c[0].T
            # TODO: check off-by-one
            x_extra = x_extra[crop_pre : -(crop_post - 1)]

            for annotator in range(data.num_annotators):
                a = np.array([annotator]).repeat(len(logreg_features))
                task = 0
                l = logreg(logreg_features, x_extra, a, task)

                l = torch.cat((l[:1].repeat(crop_pre, 1), l, l[-1:].repeat(crop_post, 1)), dim=0)
                p = torch.argmax(l, dim=-1)

                assert len(p) == x.shape[-1] + 1

                with io.BytesIO() as buffer:
                    np.savez_compressed(buffer, p.cpu().data.numpy())
                    test_predictions[g.decode("utf-8")][annotator] = buffer.getvalue()
                with io.BytesIO() as buffer:
                    np.savez_compressed(buffer, l.cpu().data.numpy().astype(np.float32))
                    test_logits[g.decode("utf-8")][annotator] = buffer.getvalue()

            for task in range(1, max(data.clf_tasks) + 1):
                a = np.array([annotator]).repeat(len(logreg_features))
                l = logreg(logreg_features, x_extra, a, task)

                l = torch.cat((l[:1].repeat(crop_pre, 1), l, l[-1:].repeat(crop_post, 1)), dim=0)
                p = torch.argmax(l, dim=-1)

                assert len(p) == x.shape[-1] + 1

                with io.BytesIO() as buffer:
                    np.savez_compressed(buffer, l.cpu().data.numpy().astype(np.float32))
                    task3_test_logits[g.decode("utf-8")][task] = buffer.getvalue()

    return test_predictions, test_logits, task3_test_logits
