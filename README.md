# Multi-Animal Behavior challenge solution outline
Author : Benjamin Wild (<b.w@fu-berlin.de>)

## Overview

The core idea of my solution was to get as much use as possible out of the limited amount of data
available using a combination of several methods:

* Feature preprocessing instead of augmentation: Use egocentric representation of the data with
  relative orientations and velocities. I added a PCA embedding of the graph consisting of
  the pairwise distances of all points of both individuals to the features, and I'm also using
  absolute spatial and temporal information in the final classification layer of the model.

* Semi-supervised learning using the InfoNCE / CPC objective. My initial attempt used a
  unsupervised CPC model pretrained on the entire dataset (train and test data combined). A linear
  classifier on these embeddings is about as good as the baseline model in Task 1. To further
  improve the accuracy, I then trained the model in a semi-supervised fashion with 75% of the
  samples in each batch sampled from the test sequences and the remaining 25% from the train
  sequences. The InfoNCE loss was computed for all samples and the task specific classification
  losses only for the samples from the train sequences
  ([Oord et al., 2019](https://arxiv.org/abs/1807.03748),
  [Chen et al, 2020](https://arxiv.org/abs/2002.05709v3)).

* Joint training of one model for all tasks: Instead of treating the three tasks separately, I opted
  to jointly train one model because I assumed that the task objectives are strongly correlated
  (i.e., the Task 1 classification task would regularize the model for the Task 3 classification
  tasks and vice versa). To that end, I use one model that extracts embeddings from the sequences
  and stack a multi-head classification layer for the three tasks on top of it.

* CV / Ensembling / Label smoothing: I relied extensively on my local cross validation. I
  group samples by their sequence ID and ensure that in each CV split, all samples from one sequence
  are either in the train or validation set (I assumed this was also how the public / private
  leaderboard worked, but based on the small differences between the public and final scores this
  is apparently not the case). This local CV pipeline had a number of benefits: a) I could evaluate
  if changes to the model significantly improved the performance without relying on and potentially
  overfitting to the public leaderboard. b) For each model trained on a CV split, I stored the
  parameters with the lowest validation loss. I then used these models in an ensemble for the
  submissions. I used label smoothing to improve the calibration of the model, thereby increasing
  the accuracy of the ensemble ([Guo et al., 2017](https://arxiv.org/abs/1706.04599),
  [Müller et al., 2020](https://arxiv.org/abs/1906.02629)).

* Dark knowledge: I tried to improve the performance by using the predictions from the ensemble as
  soft labels for the test data, i.e. to bootstrap training labels for the model using an ensemble
  of the previous version of the model. In a previous project of mine, this worked quite well and
  I hoped that it would be very beneficial for the Task 3 behaviors. In the end, this approach only
  marginally improved the performance for Task 1, and not at all for Task 2 and 3. One explanation
  for this could be that the test data for which I created those soft labels was also the data the
  model was scored on for the leaderboards. Maybe this approach would have worked better with a
  truly separate dataset for unsupervised training?
  ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531))

* Model architecture: No big surprises here. I use residual blocks with convolutional layers in a
  more or less standard ResNet/CPC architecture. The first part of the model (the embedder,
  in CPC terms) is non-causal. The second part (the CPC contexter) is a causal model. The CPC
  contexts are used in the InfoNCE objective and also in the classification head. I use LayerNorm to
  avoid potential problems of data leakage with BatchNorm and the CPC objective.

## Details

The implementation consists of two parts: a) A core library located in the folder `mabe/` containing
most of the code. b) A number of hydrogen notebooks located in the folder `hydrogen/` for EDA,
feature preprocessing, submission creation, and in general for "throwaway" / experimental code, most
of which never made it into this repository.

### Feature preprocessing

`hydrogen/features.py`, `hydrogen/getposepca.py`, `mabe/features.py`

Data augmentation is often used to encode domain knowledge in a machine learning model. Here, I
attempted to encode this prior knowledge in the model architecture and using preprocessing of
the features.

I define the orientation of a mouse as the vector from its tail to neck coordinates and compute
distances, velocities, and angles between coordinates relative to this orientation, i.e. in a
egocentric coordinate system. Angles are represented as unit vectors.

I also included a learned representation of the graph of all point-to-point euclidean distances of
all tracking points of both mice, i.e. for two mice with 7 tracking points each the graph is
stored in a 14 x 14 distance matrix. I then compute PCA features of the condensed form of these
distance matrices for all points in the train and test sets and keep the first $n$ principal
components that explain at least 95% of the total variance.

I also compute the euclidean distance to the next wall and temporal features, see `Joint training`
for more details.

### Semi-supervised learning

`mabe/model.py`

The model utilizes the CPC / InfoNCE loss, a domain-agnostic unsupervised learning objective. The
core idea is as follows: A small non-causal convolutional neural network (`Embedder`) processes the
preprocessed input sequences and returns embedded sequences $e_t$ of the same length (same number of
temporal steps). A much bigger causal convolutional network (`Contexter`) then returns contexts
$c_t$ for each temporal step. The model also jointly learns a number of linear projections $W_n$
that map contexts $c_t$ into the future $c_{t+n}$. The InfoNCE objective is minimized if the cosine
similarity from these mapped contexts $c_{t+n}$ is 1 for $e_{t+n}$ from the same sequence and 0 for
randomly sampled $e$s from other sequences. See [Oord et al.,
2019](https://arxiv.org/abs/1807.03748) for more details. Note that the loss used in my
implementation is a slight variation  of the original CPC loss (`NT-Xent`) as described in [Chen et
al., 2020](https://arxiv.org/abs/2002.05709v3).

In my implementation, I chose to use a fixed ratio (1:3) of labeled and unlabeled samples in each
batch and computed the InfoNCE objective for all samples and the classification loss only for the
labeled samples. In each batch, I sample `batch_size` sequences proportional to their length (i.e.,
a longer sequence is sampled more often) and then uniformly sample a temporal subsequence of each
sequence.

### Joint training

`mabe/training.py`

The core idea here was to share as many of the model parameters as possible for all tasks and the
CPC objective. Therefore, only the last part of the model (`MultiAnnotatorLogisticRegressionHead`)
is task-specific.

To stabilize the training, batches are sampled s.t. they always contain at least one sample for each
task, i.e. at least one sample is for the Task 0 behaviors, one for behavior-0 of Task 3, and so on.

The `MultiAnnotatorLogisticRegressionHead` consists of a LayerNorm layer, a residual block, and
final linear layers for each classification from Task 1 and 3 (i.e., a multinomial classification
layer for Task 1, and 7 binary classification layers for Task 3). The residual block has one
additional input: A learned embedding of the annotator for Task 3. This embedding is initialized as
a diagonal matrix, i.e. the embedding for annotator 0 will initially be $[1, 0, 0, 0, 0, 0]$. The
model can then learn similar embeddings for annotators with a similar style and the residual block
can modify the inputs to the classification head to match the style of each annotator. The annotator
embeddings are kept small to avoid overfitting, but I did not experiment with larger or different
kinds of embeddings.

The losses were scaled to prevent overfitting to the Task 3 classification tasks with a much lower
amount of training data.

Finally, a number of features (`extra_features`) are only concatenated in this final regression head
to avoid overfitting. I used representations of space (tanh of scaled euclidean distance to the
wall) and time (tanh of scaled timestep and (num_timesteps - timestep)) as extra features, where the
scaling factors where determined via a bayesian hyperparameter optimization (see `hydrogen/eda.py`).

### CV / Ensembling / Label smoothing

`hydrogen/create_splits.py`, `hydrogen/merge_results_alltasks.py`, `mabe/loss.py`

To be able to reliably measure if modifications to the model improved the performance without
overfitting to the public leaderboard, I created a fixed set of 32 cross-validation splits, whereby
one sequence would always be completely in either the training or the validation split. Because of
runtime and compute constraints, I usually only trained models on the first 10 splits and only
trained model on the full set of CV splits prior to a submission.

During each run, I stored the highest validation F1 scores for Task 1 and individually for each
behavior from Task 3. I also kept a copy of the parameters of the entire model with the highest
validation F1 scores for each behavior. I didn't explicitly measure the F1 scores for Task 2,
because I assumed they would be strongly correlated with the Task 1 scores. After the training, I
stored the validation scores and also predictions (logits) of the model with the highest scores in a
`TrainingResult` dataclass object and stored a compressed copy of these results to the filesystem.

Before submission, I loaded the predictions from each model and computed the average predicted
per-class probabilities. The ensemble prediction was then simply the argmax of these averaged model
predictions. Using a weighted mean with the validation F1 scores as weights did not significantly
improve the results.

Using a grouped cross validation approach worked well for Task 1 and 2, but was somewhat problematic
for Task 3, where only a small amount of sequences were available for each behavior.

### Dark knowledge

`hydrogen/features.py`, `mabe/model.py`

For Task 3, using grouped CV is problematic because only few (in one case only 1) sequences exists
per behavior. I tried to circumvent this by first training models on the CV splits, and them use the
ensemble to create additional training data, thereby effectively bootstrapping a bigger training set
from the dark knowledge ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)) of the ensemble.
Because the ensemble consists of models trained on different CV splits, it has effectively been
trained on the entire training set (all sequences).

While I think that the idea is valid and I've successfully used this approach in a previous
project, it didn't work nearly as well as I was hoping for in this challenge. I wasn't able to
definitely figure out why, but these are the two potential problems that I see: a) When training
using the dark knowledge loss, you're effectively leaking information from the validation split (via
the knowledge contained in the ensemble), thereby making strategies like early stopping on the
validation loss problematic. One way around this would be to use an additional test split, but for
most behaviors in Task 3 there are not enough sequences available to do this properly. b) I used the
dark knowledge term only for the test sequences, but these are also the sequences which get used for
scoring. Maybe the model is able to overfit to the predictions of the ensemble on the test
sequences, thereby rendering the loss term useless when trying to improve the predictions for the
test sequences.

It is possible that this approach would've worked better with a separate unlabeled dataset, or maybe
even with different loss scaling factors or by applying the dark knowledge term to the train
sequences from the other tasks, but I wasn't able to properly investigate this before the deadlines.

### Additional details

* All final models use the `MADGRAD` optimizer with cosine learning rate annealing
  [Defazio et al., 2021](https://arxiv.org/abs/2101.11075).

* Hyperparameters for the final model: All models use the default hyperparameters as defined in the
  `TrainingConfig` dataclass in `mabe/model.py` except for the Task 3 ensemble, for which
  `dark_knowledge_loss_scaler` was set to 0.

* Almost no "traditional regularisation": A small weight decay of $1e-5$ is used during
  optimization. I tried to apply dropout at various positions in the model, but it never increased
  the model performance. I also briefly experimented with augmentation, but most reasonable
  augmentations are not necessary anymore after feature preprocessing. Domain knowledge might be
  helpful in designing better augmentations.

* I used `sklearn.utils.class_weight.compute_class_weight` for all classification tasks based on the
  entire dataset (train + validation) data.

## How to reproduce results

The code assumes that all data (e.g., `train.npy`), is stored in the location defined in `ROOT_PATH`
if `mabe/config.py`. Alternatively, the environment flag `MABE_ROOT_PATH` can be set to override
this config variable.

0. Optional: Use the `hydrogen/getposepca.py` notebook to get the PCA model for the point-to-point
distance graph features based on all data points from all three tasks.

1. Run feature preprocessing notebook: `hydrogen/features.py`. This will create a hdf5 file with
the preprocessed sequences for all three tasks and the test data. Note: To reproduce the final
results, you need to have a previously trained ensemble of models for the dark knowledge loss terms
and a pretrained PCA for the relative positions of the tracking points. The improvements from
these approaches are marginal and could also be ignored.

2. Create cross validation splits: I initially created a set of 32 fixed CV splits to be able to
reliably test the effects of modifications to the model. These splits can be created using the
`hydrogen/create_splits` notebook.

3. Train models using training scripts `scripts/train.py`: This command line script trains a batch
on model on the given CV splits and returns the cross validated F1 scores. All training
hyperparameters defined in the `TrainingConfig` dataclass in `mabe/training.py` can be set using
command line flags. Example:

    > `./train.py --model_name "ensemble_0to5" --device "cuda:0" --feature_path features.hdf5
    --weight_decay=1e-5 --from_split 0 --to_split 5`

4. Optional: Use this ensemble to bootstrap training data for step 1 using the
`hydrogen/features.py` notebook.

5. Create final submission: Use the notebook `hydrogen/merge_results_alltasks.py` to load the
ensemble predictions, average them, and create the submissions for the three tasks.

## Final remarks

I tried many things which turned out to not help much. The codebase is therefore somewhat
convoluted and could be improved significantly if only the core functionality were desired. If I
were to use such a model in production, I would only use good feature preprocessing with as much
domain knowledge as possible, and utilize semi-supervised training using the InfoNCE objective. If
accuracy was absolutely critical, I would also use an ensembling approach.

Here are a couple of ideas that might further improve the results:

* Class-Balanced Loss ([Cui et al, 2019](https://arxiv.org/abs/1901.05555)).

* Proper hyperparameter optimization for CPC (Batch size, embedding size, ...). There are also some
  recent papers that describe improvements to CPC, in particular w.r.t. the selection of negative
  samples for the InfoNCE objective
  ([e.g., Robinson et al., 2021](https://arxiv.org/abs/2010.04592)).

* Train model on task 1 and 2, and only fine-tune final classification layer for task 3: This
approach performed well in my local CV, but for some reason not at all on the leaderboard. I don't
know why.

* Extract embeddings from raw video data instead of pose tracking. I think there's enough data available
that such an approach might be feasible here, in particular in the semi-supervised setting.
