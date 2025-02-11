<!-- ### Embedding_dim = 10; Hidden_dim = 128 (without much data preprocessing)
- 3-gram LM: 224249431451994.22
- 5-gram LM: 401489.63280091883

### Embedding_dim = 50; Hidden_dim = 256
- 3-gram LM: 133890959252.97466
- 5-gram LM: 89983578187867.2

### Embedding_dim = 128; Hidden_dim = 512
- 3-gram LM: 6.591122514969873e+22
- 5-gram LM: inf -->

## Feed Forward Neural Network - Pride and Prejudice
- 3-gram LM on training dataset: 47.39556121826172
- 5-gram LM on training dataset: 39.426231384277344

- 3-gram LM on test dataset: 4.497087168459571e+16
- 5-gram LM on test dataset: 5.3836463290607206e+17

## Vanilla Recurrent Neural Network - Pride and Prejudice
- 3-gram LM on training dataset: 59.308475494384766
- 5-gram LM on training dataset: 59.16526794433594

- 3-gram LM on test dataset: 333128.5625
- 5-gram LM on test dataset: 396490.96875

## Long Short-Term Memory Language Model - Pride and Prejudice
- 3-gram LM on training dataset: 59.72645568847656
- 5-gram LM on training dataset: 56.50226974487305

- 3-gram LM on test dataset: 392291.96875
- 5-gram LM on test dataset: 946297.3125

## Vanilla Recurrent Neural Network - Ulysses
- 3-gram LM on training dataset: 87.85037231445312
- 5-gram LM on training dataset: 82.12122344970703

- 3-gram LM on test dataset: 417948.4375
- 5-gram LM on test dataset: 342903.6875

## Feed Forward Neural Network - Ulysses
- 3-gram LM on training dataset: 70.0067367553711
- 5-gram LM on training dataset: 64.32366180419922

- 3-gram LM on test dataset: 4.67374832345233e+22
- 5-gram LM on test dataset: 5.833692885433286e+25

## Long Short-Term Memory Language Model - Ulysses
- 3-gram LM on training dataset: 94.251220703125
- 5-gram LM on training dataset: 85.57880401611328

- 3-gram LM on test dataset: 1521657.25
- 5-gram LM on test dataset: 2536408.5


# Observations

## Ranking Based on Performance

1. Best Training Performance (lowest perplexity):

Old (Linear Interpolation, 1-gram): ~35 train perplexity for Pride and Prejudice is the best in terms of raw numbers.
2. Neural Models (Pride and Prejudice Training):

FFNN (5-gram): ~39.43 (appears best on training but overfits badly)
FFNN (3-gram): ~47.40
RNN & LSTM: ~59
3. Generalization / Test Performance:

Among neural models, RNN demonstrates test perplexities that are orders of magnitude lower than those of the FFNN model.
The LSTM shows test perplexities comparable to or slightly higher than the RNN, indicating that for longer sentences (or in less controlled test environments), the recurrence (especially vanilla RNN) generalizes better than the FFNN.
4. Corpus Differences:

For Ulysses, all models tend to have higher perplexity values compared to Pride and Prejudice, and the ordering remains similar: FFNN has the lowest training perplexity but terrible test performance, while RNN and LSTM have moderate (but still high) test perplexities.

## Detailed Analysis and Insights

### Overfitting in FFNN Models:

- The FFNN models show promising low perplexity on training data but very high perplexity on test data. This suggests that while fixed-size context (n‑grams) may be learned well in a controlled training set, the lack of sequential processing limits the model’s ability to generalize for longer sentences or unseen word patterns.

### RNN vs. LSTM for Longer Sentences:

- Vanilla RNNs and LSTMs both model longer-term dependencies. However, in these results, RNNs tend to perform slightly better on the test set than LSTMs.
- The LSTM’s gating mechanisms are designed for scenarios where long-range dependencies are crucial, yet they might require more data or better tuning to fully leverage that strength.
- In the context of these experiments, the simpler RNN may be more robust for longer sentences, likely because the increased complexity of LSTM might be harder to train or overfit if not adequately regularized.

### Effect of n-Gram Size on FFNN Models:

- Using a 5-gram input for the FFNN tends to lower the training perplexity compared to a 3-gram input, as more context is provided.
- However, the test performance drastically deteriorates. This indicates that while a higher n may help capture more local context during training, it also increases the risk of overfitting because the model becomes too specialized to particular fixed-length contexts seen during training, making it less robust to variations in real-world/test sentences.

### Classical Smoothing Models vs. Neural Models:

- The older classical methods, particularly those using smoothing and interpolation, generally yield lower perplexity values compared to the raw neural models.
- This could be due to the fact that classical methods, when tuned well, have simpler assumptions and built-in mechanisms for handling unseen data.
- Neural models offer flexibility and the ability to learn rich representations but are more sensitive to training data size, hyperparameter settings, and regularization techniques.

## Ranking by Test Performance:

- Among neural models, Vanilla RNNs tend to generalize better than both FFNNs and LSTMs for longer sentences under the current setup.
For FFNN models, while a 5-gram context reduces training perplexity, it leads to severe overfitting and thus very poor test performance.
Classical n-gram models with smoothing/interpolation still provide competitive (and often better) perplexity scores in controlled experiments, especially when generalizing to unseen data.
Which Model Performs Better for Longer Sentences?

- The Vanilla RNN performs relatively better on longer sentences. Its recurrent structure allows it to maintain a hidden state that captures sequential dependencies over time, whereas the FFNN’s fixed context window limits its ability to integrate information beyond that window.
Impact of n-Gram Size in FFNNs:

- Increasing n (e.g., from 3-gram to 5-gram) generally improves training performance by providing more context. However, an excessively high n aggravates overfitting, leading to drastically higher perplexity on test data. A balance must be struck between context size and generalization capacity.