# Project 4: NLP & Sentiment Analysis

DecodeLabs Data Science Industrial Training — Batch 2026

## Overview
This project builds an end-to-end pipeline that reads unstructured text (movie/product reviews) and classifies it as **Positive** or **Negative** sentiment.

## Pipeline
1. **Text Pre-Processing** — HTML tag removal, lowercasing, punctuation/digit removal, tokenization, stop-word removal, lemmatization (NLTK).
2. **Vectorization** — TF-IDF (unigrams + bigrams, top 5,000 features).
3. **Classification** — Multinomial Naive Bayes and Linear SVM, trained and compared.
4. **Evaluation** — Accuracy, precision, recall, F1-score, confusion matrices.
5. **Inference** — `predict_sentiment()` function to classify any new review.

## Dataset
**IMDB Dataset of 50,000 Movie Reviews** (25,000 positive / 25,000 negative) — the standard benchmark dataset for binary sentiment classification, originally from Stanford's Large Movie Review Dataset (Maas et al., ACL 2011).

The notebook loads it directly from a public GitHub-hosted CSV mirror, so it runs on Google Colab or locally with **no manual download/upload required**:
```
https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv
```
Columns: `review` (raw review text), `sentiment` (`positive` / `negative`).

## Results (on full 50k dataset, 80/20 split)
| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Naive Bayes | 0.862 | 0.86 | 0.86 | 0.86 |
| Linear SVM | 0.886 | 0.89 | 0.89 | 0.89 |

Linear SVM outperforms Naive Bayes, consistent with typical results for TF-IDF + SVM on text classification.

## Files
- `Project4_NLP_Sentiment_Analysis.ipynb` — full notebook (Colab-ready)
- `sentiment_model.pkl` — trained best model (Linear SVM)
- `tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer

## How to Run
1. Open the notebook in Google Colab or Jupyter.
2. Run all cells top to bottom — the dataset downloads automatically.
3. Full preprocessing + training takes ~2-3 minutes on Colab CPU.

## Possible Extensions
- Hyperparameter tuning via GridSearchCV
- Word2Vec/GloVe embeddings for comparison
- Transformer-based model (BERT) benchmark
- Deploy as a simple web app / API
