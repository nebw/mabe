import collections
import io

import numpy as np
import scipy
import scipy.special
import torch
from fastprogress.fastprogress import force_console_behavior

import mabe
import mabe.config
import mabe.loss
import mabe.model
import mabe.training
import mabe.util

# %%
master_bar, progress_bar = force_console_behavior()

# %%
results_files = (
    "training_results_2021-05-06 03:44:12.966718_final_ensemble5_0to2_0.886.pt",
    "training_results_2021-05-06 09:21:07.001650_final_ensemble5_2to4_0.816.pt",
    "training_results_2021-05-06 15:31:40.573751_final_ensemble5_4to6_0.844.pt",
    "training_results_2021-05-06 21:23:28.094687_final_ensemble5_6to8_0.747.pt",
    "training_results_2021-05-06 03:42:25.017044_final_ensemble5_8to10_0.892.pt",
    "training_results_2021-05-06 09:07:51.314619_final_ensemble5_10to12_0.829.pt",
    "training_results_2021-05-06 09:41:19.682021_final_ensemble5_10to12_0.845.pt",
    "training_results_2021-05-06 15:12:38.257900_final_ensemble5_12to14_0.846.pt",
    "training_results_2021-05-06 20:45:42.821152_final_ensemble5_14to16_0.843.pt",
    "training_results_2021-05-06 03:27:10.472722_final_ensemble5_16to18_0.780.pt",
    "training_results_2021-05-06 16:55:01.388127_final_ensemble5_20to22_0.776.pt",
    "training_results_2021-05-07 01:04:17.931439_final_ensemble5_22to24_0.861.pt",
    "training_results_2021-05-06 03:36:34.616866_final_ensemble5_24to26_0.874.pt",
    "training_results_2021-05-06 10:22:00.194306_final_ensemble5_26to28_0.838.pt",
    "training_results_2021-05-06 17:00:43.363940_final_ensemble5_28to30_0.764.pt",
    "training_results_2021-05-06 23:41:20.680606_final_ensemble5_30to32_0.855.pt",
)

# %%
test = np.load(mabe.config.ROOT_PATH / "test-release.npy", allow_pickle=True).item()

# %%
def get_task3_scores(result):
    mean_task3_score = np.mean(
        list(map(lambda v: np.max(np.array(v)), list(result.clf_val_f1s.values())[1:4]))
        + list(map(lambda v: np.max(np.array(v)), list(result.clf_val_f1s.values())[5:]))
    )
    return [
        (max(v) if v[0] is not None else mean_task3_score) for k, v in result.clf_val_f1s.items()
    ]


# %%
results = []
scores_list = []
for filename in progress_bar(results_files):
    result_batch = torch.load(mabe.config.ROOT_PATH / filename)
    results += list(map(lambda r: r.task3_test_logits, result_batch))
    scores_list += [np.stack(list(map(get_task3_scores, result_batch)))]

scores = np.concatenate(scores_list)
scores = scores.T

np.percentile(scores, [10, 50, 90], axis=0).mean(axis=1)

# %%
sample_submission = np.load(
    mabe.config.ROOT_PATH / "sample-submission-task3.npy", allow_pickle=True
).item()

# %%
submission: dict = collections.defaultdict(dict)
for behavior_key in progress_bar(sample_submission.keys()):
    behavior_idx = int(behavior_key[-1])

    for key, sequence in test["sequences"].items():
        preds = np.stack(
            list(
                map(
                    lambda s: np.load(io.BytesIO(s[key][behavior_idx + 1]))["arr_0"],
                    results,
                )
            )
        )

        """
        preds = np.argmax(
            np.average(scipy.special.softmax(preds, -1), weights=scores[behavior_idx + 1], axis=0),
            axis=-1,
        )
        """
        preds = np.argmax(np.mean(scipy.special.softmax(preds, -1), axis=0), axis=-1)
        submission[behavior_key][key] = preds

# %%
def validate_submission_task3(submission, sample_submission):
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
if validate_submission_task3(submission, sample_submission):
    np.save(mabe.config.ROOT_PATH / "task3_submission.npy", submission)

# %%
sample_submission = np.load(
    mabe.config.ROOT_PATH / "sample-submission.npy", allow_pickle=True
).item()

# %%
merged_submission = {}
for key in progress_bar(sample_submission.keys()):
    annotator_id = 0
    preds = np.stack(
        list(map(lambda s: np.load(io.BytesIO(s[key][annotator_id]))["arr_0"], results))
    )
    preds = np.argmax(np.average(scipy.special.softmax(preds, -1), weights=scores, axis=0), axis=-1)

    merged_submission[key] = preds

# %%
def validate_submission_task1(submission, sample_submission):
    if not isinstance(submission, dict):
        print("Submission should be dict")
        return False

    if not submission.keys() == sample_submission.keys():
        print("Submission keys don't match")
        return False

    for key in submission:
        sv = submission[key]
        ssv = sample_submission[key]
        if not len(sv) == len(ssv):
            print(f"Submission lengths of {key} doesn't match")
            return False

    for key, sv in submission.items():
        if not all(isinstance(x, (np.int32, np.int64, int)) for x in list(sv)):
            print(f"Submission of {key} is not all integers")
            return False

    print("All tests passed")
    return True


# %%
if validate_submission_task1(merged_submission, sample_submission):
    np.save(mabe.config.ROOT_PATH / "task1_submission.npy", merged_submission)


# %%
merged_submission = {}
for key, sequence in test["sequences"].items():
    annotator_id = sequence["annotator_id"]
    preds = np.stack(
        list(map(lambda s: np.load(io.BytesIO(s.test_logits[key][annotator_id]))["arr_0"], results))
    )
    preds = np.argmax(scipy.special.softmax(preds, -1).mean(axis=0), axis=-1)

    merged_submission[key] = preds

# %%
sample_submission = np.load(
    mabe.config.ROOT_PATH / "sample-submission-task2.npy", allow_pickle=True
).item()


def validate_submission_task2(submission, sample_submission):
    if not isinstance(submission, dict):
        print("Submission should be dict")
        return False

    if not submission.keys() == sample_submission.keys():
        print("Submission keys don't match")
        return False

    for key in submission:
        sv = submission[key]
        ssv = sample_submission[key]
        if not len(sv) == len(ssv):
            print(f"Submission lengths of {key} doesn't match")
            return False

    for key, sv in submission.items():
        if not all(isinstance(x, (np.int32, np.int64, int)) for x in list(sv)):
            print(f"Submission of {key} is not all integers")
            return False

    print("All tests passed")
    return True


# %%
if validate_submission_task2(merged_submission, sample_submission):
    np.save(mabe.config.ROOT_PATH / "task2_submission.npy", merged_submission)
