# Sentiment Analysis with RNN, BiLSTM, and GRU


## 📌 Overview
This project performs **binary sentiment analysis** on a large text corpus (loaded from compressed `.bz2` files with labels like `__label__1` / `__label__2`).  
The notebook trains and compares three deep learning models:
- **Baseline SimpleRNN**
- **Bidirectional LSTM (BiLSTM)**
- **GRU**

It includes data loading, text cleaning, tokenization, sequence padding, model training with early stopping, and evaluation using accuracy and a scikit-learn **classification report** (precision/recall/F1).

---

## 🗂️ Dataset & Sampling
- **size**: 40,00,000 reviews
- due to hardware limitaions taken a small subset form the data
- **Train**: 15,000 reviews from `train.ft.txt.bz2`
- **Test**: 5,000 reviews from `test.ft.txt.bz2`
- Labels are extracted via regex from lines like `__label__1 some review text...`
- Text is cleaned and labels mapped to binary (0/1).


---

## 🧹 Preprocessing
- `Tokenizer(num_words=max_word)` with **`max_word=1000`**
- `max_seq_len=100`
- `pad_sequences` to unify input length
- Custom `clean_txt()` for basic text cleaning
- Train/validation split is done via passing `validation_data=(X_test, y_test)`

---

## 🧠 Models

### 1) Baseline: SimpleRNN (stacked)
```python
Embedding(input_dim=max_word, output_dim=300, input_length=max_seq_len)
SimpleRNN(128, return_sequences=True, dropout=0.3)
LayerNormalization()
SimpleRNN(128, dropout=0.3)
LayerNormalization()
Dropout(0.3)
Dense(64, activation='relu')
Dropout(0.2)
Dense(1, activation='sigmoid')
```
**Training:** `epochs=100`, `batch_size=512`, `optimizer=Adam(5e-5)`, **EarlyStopping** on `val_loss` (patience=5)

### 2) Bidirectional LSTM
```python
Embedding(input_dim=max_word, output_dim=300, input_length=max_seq_len)
Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.3))
LayerNormalization()
Dropout(0.3)
Bidirectional(LSTM(128, recurrent_dropout=0.3))
LayerNormalization()
Dropout(0.3)
Dense(64, activation='relu')
Dropout(0.2)
Dense(1, activation='sigmoid')
```
**Training:** `epochs=50`, `batch_size=512`, `optimizer=Adam(5e-5)`, **EarlyStopping**

### 3) GRU
```python
Embedding(input_dim=max_word, output_dim=300, input_length=max_seq_len)
Bidirectional(GRU(128, return_sequences=True, reset_after=True, recurrent_dropout=0.3))
LayerNormalization()
Dropout(0.3)
Bidirectional(GRU(128, reset_after=True, recurrent_dropout=0.3))
LayerNormalization()
Dropout(0.3)
Dense(64, activation='relu')
Dropout(0.2)
Dense(1, activation='sigmoid')
```
**Training:** `epochs=50`, `batch_size=512`, `optimizer=Adam(5e-5)`, **EarlyStopping**

> All models use **binary cross-entropy** loss and report the **accuracy** metric.

---

## 📈 Results (from notebook outputs)

| Model        | Best Val Accuracy (approx.) | Test Classification Report (Weighted F1) |
|--------------|-----------------------------|------------------------------------------|
| SimpleRNN    | ~0.78–0.79                  | —                                        |
| BiLSTM       | ~0.82–0.83                  | **0.83** on 5,000 test samples           |
| GRU          | ~0.82–0.83                  | **0.82–0.83** on 5,000 test samples      |

Additional training logs show validation accuracy peaking around **0.824–0.826** for BiLSTM/GRU, with stable validation loss under early stopping. The SimpleRNN baseline improves more slowly and underperforms the gated variants.


---

## 🔬 Comparison & Notes
- **BiLSTM and GRU** both clearly outperform the **SimpleRNN** baseline on this subset.
- The vocabulary cap (`num_words=1000`) and short `max_seq_len=100` likely bottleneck performance.
- Even classical baselines (e.g., **Logistic Regression + TF‑IDF**) can be strong; Transformers (**DistilBERT/BERT**) often exceed **90%** on similar polarity datasets with enough data.

---

## 🚀 How to Run
1. Install deps:
   ```bash
   pip install tensorflow scikit-learn pandas numpy matplotlib
   ```
2. Place the compressed data files in the expected paths:
   - `train.ft.txt.bz2`
   - `test.ft.txt.bz2`
3. Open the notebook and run cells in order.
4. Optional: adjust **`max_word`** and **`max_seq_len`** for better accuracy.

---

## 🔎 Inference Helper
A convenience function is included in the notebook:
```python
def predict_sentiment(text):
    # Uses the tokenizer + padding + model to return "Positive"/"Negative"
    ...
```
Replace `model_lstm` with the model you want to use (e.g., `model_GRU`).

---

## 💡 Future Work / Improvements
1. **Increase vocabulary & sequence length**: try `num_words=20_000+`, `max_seq_len=200–256`.
2. **Use pre-trained embeddings**: GloVe/CommonCrawl/FastText; set `trainable=True` after warm-start.
3. **Text normalization**: better cleaning for emojis, hashtags, URLs, contractions; consider lemmatization.
4. **Regularization & training**: tune `dropout`, try `SpatialDropout1D`, add **learning-rate schedules** and **weight decay**.
5. **Classical baselines**: TF‑IDF + Logistic Regression/SVM as a quick, strong baseline.
6. **CNN/TextCNN**: 1D convs with multiple kernel sizes often do well on short texts.
7. **Transformers**: fine‑tune **DistilBERT/BERT/ALBERT**; with 20k+ samples you can often reach **>90%** accuracy.
8. **Data scale**: train on more than 15k/5k samples; the full dataset is much larger.
9. **Threshold tuning**: tune the sigmoid decision threshold using validation ROC to balance precision/recall.
10. **Error analysis**: inspect misclassified examples; build a confusion matrix; slice by text length, punctuation, etc.
11. **Reproducibility**: set random seeds, persist tokenizer and model (`.json`/`.h5`), and add a `requirements.txt`.

---




