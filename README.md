# Fake News Detection — BiLSTM vs. Multiple Pretrained Transformers

A binary text classification project comparing a from-scratch Bidirectional LSTM
against four pretrained transformer models (two ALBERT variants, two BERT variants)
fine-tuned on the same fake-news dataset, under identical training settings.

## Overview

Given a news article (title, body text, subject, and date), the goal is to predict
whether it is real or fake. Five models are trained and compared under the same
conditions to see how much pretraining — and which pretrained architecture — actually
matters on this task.

## Dataset

- **Source:** A fake/real news dataset (6,000 articles, balanced 3,000 / 3,000 across
  the two classes). Columns: `title`, `text`, `subject`, `date`, `label`.
- **Preprocessing:** all four text fields lowercased, whitespace/newline-normalized,
  and concatenated into a single input string per article.
- **Split:** 80% train / 20% test, `random_state=1`.

## Models compared

| Model                       | Type                       | Params   |
|------------------------------|----------------------------|----------|
| BiLSTM (from scratch)         | Embedding + BiLSTM + LSTM  | ~1.7M    |
| ALBERT-base (fine-tuned)      | `albert_base_en_uncased`   | ~11.7M   |
| BERT-tiny (fine-tuned)        | `bert_tiny_en_uncased`     | ~4.4M    |
| ALBERT-large (fine-tuned)     | `albert_large_en_uncased`  | ~17.7M   |
| BERT-medium (fine-tuned)      | `bert_medium_en_uncased`   | ~41.4M   |

All four pretrained models were fine-tuned end-to-end with identical settings: Adam
at 5e-5, batch size 2, up to 5 epochs with early stopping (patience 2) and
learning-rate reduction on plateau. The BiLSTM was trained from scratch on tokenized,
padded sequences (max length 995) with Adam at a much higher learning rate (0.004),
since it has no pretrained weights to preserve.

## Results

| Model              | Test Accuracy | Approx. time/epoch |
|---------------------|---------------|----------------------|
| BiLSTM (scratch)     | ~49.1%        | ~11s                 |
| ALBERT-base          | ~48.0%        | ~400s+               |
| **BERT-tiny**        | **~99.9%**    | ~10–20s              |
| ALBERT-large         | ~52.0%        | ~1,270s (~21 min)    |
| **BERT-medium**      | **~99.75%**   | ~85–210s             |

*(Update these numbers if you re-run the notebook — they reflect the run captured
in the original notebook outputs.)*

## Key finding

The result here is more nuanced than "pretraining beats training from scratch." Both
**BERT** variants converged to near-perfect accuracy within 1–2 epochs, while both
**ALBERT** variants — despite also being pretrained, and despite one of them (large)
having more parameters than either BERT model — never moved off chance-level accuracy,
matching the from-scratch BiLSTM almost exactly. Since all four pretrained models used
the identical fine-tuning recipe (same learning rate, batch size, and epoch budget),
the gap tracks the *architecture family*, not model size or "pretrained vs. not."

This is a useful, honest result rather than an inconvenience to hide: it suggests
ALBERT's cross-layer parameter sharing may need a different learning-rate schedule
or warmup to fine-tune stably under this exact setup, and it's a reminder that
"pretrained" alone doesn't guarantee easy convergence — the specific architecture and
hyperparameter match still matters. Worth investigating further (not yet tried here):
a learning-rate warmup schedule for the ALBERT models, a larger batch size, or a
lower peak learning rate.

Training time is also a meaningful axis on its own: BERT-tiny reached the best
accuracy-to-compute ratio by a wide margin, training in seconds per epoch versus
ALBERT-large's ~21 minutes per epoch for a dramatically worse result — a genuinely
useful practical takeaway for anyone choosing a model under a compute budget.

## Tech stack

- Python, NumPy, pandas
- TensorFlow / Keras, keras-hub (for the ALBERT/BERT presets)
- scikit-learn (splitting, evaluation)
- matplotlib / seaborn (visualization)

## How to run

**BiLSTM model:** runs fine locally with a standard TensorFlow install.

**ALBERT/BERT fine-tuning:** requires meaningful GPU memory, disk space for the
pretrained weights, and patience for the larger models (ALBERT-large took roughly
20+ minutes per epoch on a single GPU). Recommended to run this section on
[Google Colab](https://colab.research.google.com/) rather than locally.

```bash
pip install -r requirements.txt
jupyter notebook notebooks/fake_news_bilstm_vs_transformers.ipynb
```

**Note:** the committed notebook was executed end-to-end on Colab so that every
model summary and training log is preserved as static output — no need to re-run it
to see the full results; just open the `.ipynb` file directly on GitHub or locally.

## Project structure

```
fake-news-bilstm-vs-transformers/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── news_dataset.csv
└── notebooks/
    └── fake_news_bilstm_vs_transformers.ipynb
```

## License

Data used under its original public license/terms; code in this repository is
available under the MIT License (see LICENSE).
