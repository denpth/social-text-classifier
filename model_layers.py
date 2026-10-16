import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom Layer: Masked Global Average Top K Pooling Layer
@tf.keras.utils.register_keras_serializable(package="portfolio")
class MaskedGlobalAverageTopKPooling1D(Layer):
    def __init__(self, k=3, **kwargs):
        super(MaskedGlobalAverageTopKPooling1D, self).__init__(**kwargs)
        self.k = k

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)
            mask = tf.expand_dims(mask, axis=-1)
            neg_inf = tf.constant(-1e9, dtype=inputs.dtype)
            inputs = inputs + (1.0 - mask) * neg_inf
        x = tf.transpose(inputs, perm=[0, 2, 1])
        top_k_values, _ = tf.math.top_k(x, k=self.k)
        valid_mask = tf.cast(top_k_values > -1e8, inputs.dtype)
        clean_values = top_k_values * valid_mask
        counts = tf.reduce_sum(valid_mask, axis=-1)
        return tf.reduce_sum(clean_values, axis=-1) / tf.maximum(counts, 1.0)

    def get_config(self):
        config = super(MaskedGlobalAverageTopKPooling1D, self).get_config()
        config.update({"k": self.k})
        return config
        
    def compute_mask(self, inputs, mask=None):
        return None
