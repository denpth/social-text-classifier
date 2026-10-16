# Social Text Classification Pipeline

## Project context

Developed during Western Governors University coursework. This repository is a record of my implementation and learning, not a client engagement, deployed operational system, or claim of a new algorithm. Supplied requirements, starter components, datasets, and third-party libraries are distinct from my own implementation. Where their exact provenance or redistribution terms have not been established, they are not represented as my original work or as cleared for public release.


An NLP project for preprocessing social-media-style text and assigning one of three moderation-oriented output categories. The repository includes the model-development notebook, saved model artifacts, preprocessing snapshots, evaluation data, and a standalone inference script.

## Inference pipeline

`inference.py` shares preprocessing with the model-development notebook. Saved-model loading and prediction parity are verified by the retained integration record and regression fixtures. This checks implementation consistency; historical vocabulary-before-split and contraction-handling limitations remain. Processing includes:

- URL and user-handle normalization
- Emoticon and emoji normalization
- Repeated-character and punctuation handling
- Contraction expansion
- Numeric normalization
- Tweet-aware tokenization
- Noun and verb lemmatization
- Vocabulary lookup and sequence padding
- Three-class probability output

## Repository contents

- `D803.ipynb` — model-development notebook
- `inference.py` — reusable inference entry point
- `Models/` — saved model artifact
- `snapshot_data/` — saved preprocessing/tokenizer artifacts
- `KerasTuner/` — tuning outputs
- `final_df.csv` and `processed_data.csv` — prepared datasets
- `missed_words.csv` — vocabulary-analysis artifact

## Run inference

The script expects the saved model at `Models/model.keras` and the serialized word index at `snapshot_data/word_index.pickle`.

Install the Python dependencies used by the script, including TensorFlow, NumPy, NLTK, and `emoji`, then run:

```bash
python inference.py "An example sentence."
```

The intended output is a predicted category and per-class probabilities. The notebook and predictor now share token cleaning and post-padding/truncation. Regression tests cover short, long, unknown, and empty sequences plus a saved custom-layer fixture and prediction parity. The historical model and matching vocabulary also load successfully and match notebook-style direct evaluation on seven synthetic text cases. Model accuracy remains a separate, unverified question.

## Topics demonstrated

Text normalization, tokenization, lemmatization, vocabulary management, sequence modeling, Keras model persistence, hyperparameter tuning, and the remaining work needed for reproducible inference.

## October 2026 integration correction

`preprocessing.py` is the shared contract and `model_layers.py` registers the custom pooling layer. Run `python -c "from preprocessing import ensure_resources; ensure_resources()"` once to install WordNet resources, then `python -m unittest discover -s tests`. Dependencies: TensorFlow, numpy, nltk, emoji.

The previous standalone predictor used pre-padding/pre-truncation while training used post/post. It now matches training. Tests cover both an independently constructed tiny model fixture and the historical model with its saved vocabulary. The historical artifact test checks post/post tokens and prediction parity on seven synthetic inputs; it does not establish classification quality. Legacy cleaning order and zero-valued unknown tokens are preserved to avoid silently changing existing artifacts. Apostrophe/contraction handling and training-only vocabulary fitting should be versioned with a new training run before changing them. The historical vocabulary was built before splitting and that evaluation limitation remains. Only load trusted local model and pickle artifacts.

## Public portfolio release

This is a publication copy of my WGU coursework implementation and subsequent revisions. The original development history is retained privately. Assessment documents, student identifiers, and training-data snapshots are excluded from this public history. Notebook outputs are cleared; recorded evaluation artifacts are retained separately where available. The academic origin and third-party dependencies remain attributed.

Training CSVs, training arrays, and tuner checkpoints are excluded. The saved inference model, vocabulary mapping, code, and regression fixtures remain available. The notebook requires separately obtained, appropriately licensed data for retraining.
