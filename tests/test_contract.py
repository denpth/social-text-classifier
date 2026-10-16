import json
import pickle
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import clean_text, encode_tokens
from model_layers import MaskedGlobalAverageTopKPooling1D
from inference import ToxicityPredictor

class ContractTests(unittest.TestCase):
    def test_cleaning_matches_recorded_legacy_inputs(self):
        cases=json.loads((Path(__file__).parent/"legacy-cleaning-golden.json").read_text())
        for case in cases:
            self.assertEqual(clean_text(case["text"]),case["tokens"])

    def test_post_padding_truncation_unknown_and_empty(self):
        vocab = {"a":1,"b":2,"c":3,"d":4}
        cases = [[],["a"],["a","missing","b"],["a","b","c","d"]]
        for tokens in cases:
            expected = pad_sequences([[vocab.get(t,0) for t in tokens]],maxlen=3,padding="post",truncating="post")[0]
            np.testing.assert_array_equal(encode_tokens(tokens,vocab,3), expected)
        self.assertEqual(encode_tokens(["a","b","c","d"],vocab,3),[1,2,3])
        self.assertEqual(encode_tokens(["a"],vocab,3),[1,0,0])
        with self.assertRaises(ValueError): encode_tokens([],vocab,0)

    def test_historical_model_matches_notebook_contract(self):
        root=Path(__file__).resolve().parents[1]
        predictor=ToxicityPredictor(root/"Models/model.keras",root/"snapshot_data/word_index.pickle")
        cases=json.loads((Path(__file__).parent/"legacy-cleaning-golden.json").read_text())
        texts=[case["text"] for case in cases]
        expected=pad_sequences([[predictor.word_index.get(token,0) for token in case["tokens"]] for case in cases],maxlen=predictor.max_len,padding="post",truncating="post")
        np.testing.assert_array_equal(predictor.prepare(texts),expected)
        outputs=predictor.model(expected,training=False).numpy()
        actual=predictor.predict(texts)
        np.testing.assert_allclose([list(row["probabilities"].values()) for row in actual],outputs,atol=1e-6)

    def test_saved_custom_layer_and_prediction_parity(self):
        texts=["hello world", "hello", "", "hello world hello world hello world"]
        vocab={word:i+1 for i,word in enumerate(sorted(set(sum([clean_text(t) for t in texts],[]))))}
        train_inputs=np.asarray([encode_tokens(clean_text(t),vocab,5) for t in texts],dtype=np.int32)
        tf.keras.utils.set_random_seed(42)
        model=tf.keras.Sequential([tf.keras.Input(shape=(5,),dtype="int32"),
            tf.keras.layers.Embedding(len(vocab)+1,4,mask_zero=True),
            MaskedGlobalAverageTopKPooling1D(k=2),tf.keras.layers.Dense(3,activation="softmax")])
        expected=model(train_inputs,training=False).numpy()
        with tempfile.TemporaryDirectory() as folder:
            model_path=Path(folder)/"fixture.keras"; words=Path(folder)/"vocab.pickle"
            model.save(model_path); words.write_bytes(pickle.dumps(vocab))
            predictor=ToxicityPredictor(model_path,words)
            np.testing.assert_array_equal(predictor.prepare(texts),train_inputs)
            actual=predictor.predict(texts)
            np.testing.assert_allclose([list(x["probabilities"].values()) for x in actual],expected,atol=1e-6)
            self.assertEqual(predictor.predict([]),[])
            with self.assertRaises(ValueError): ToxicityPredictor(model_path,words,max_len=6)

if __name__=="__main__": unittest.main()
