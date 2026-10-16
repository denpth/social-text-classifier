"""Inference for trusted local model/vocabulary artifacts; not a moderation service."""
import pickle
import numpy as np
import tensorflow as tf
from preprocessing import clean_text, encode_tokens
from model_layers import MaskedGlobalAverageTopKPooling1D

class ToxicityPredictor:
    def __init__(self, model_path, word_index_path, max_len=None):
        self.model = tf.keras.models.load_model(model_path, compile=False,
            custom_objects={"MaskedGlobalAverageTopKPooling1D": MaskedGlobalAverageTopKPooling1D})
        # Only load pickle files from trusted local training runs.
        with open(word_index_path, "rb") as stream:
            self.word_index = pickle.load(stream)
        inferred = self.model.input_shape[1]
        if max_len is None:
            max_len = inferred
        if not isinstance(max_len, int) or isinstance(max_len, bool) or max_len <= 0:
            raise ValueError("A positive max_len is required for a dynamic input shape")
        if inferred is not None and max_len != inferred:
            raise ValueError("max_len does not match the model input shape")
        self.max_len = max_len

    def prepare(self, texts):
        return np.asarray([encode_tokens(clean_text(text), self.word_index, self.max_len)
                           for text in texts], dtype=np.int32)

    def predict(self, text_input):
        single = isinstance(text_input, str)
        texts = [text_input] if single else list(text_input)
        if not texts:
            return []
        predictions = self.model.predict(self.prepare(texts), verbose=0)
        if predictions.shape != (len(texts), 3):
            raise ValueError("Expected three class probabilities per input")
        classes = ["Safe", "Offensive", "Hate"]
        results = [{"text": text, "label": classes[int(np.argmax(pred))],
                    "confidence": float(np.max(pred)),
                    "probabilities": dict(zip(classes, map(float, pred)))}
                   for text, pred in zip(texts, predictions)]
        return results[0] if single else results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--model", default="Models/model.keras")
    parser.add_argument("--vocabulary", default="snapshot_data/word_index.pickle")
    args = parser.parse_args()
    print(ToxicityPredictor(args.model, args.vocabulary).predict(args.text))
